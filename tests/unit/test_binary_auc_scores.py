"""Binary rank metrics accept real decision scores without inventing probabilities."""
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from iaml.actionables.normalize.act_standard_scaler import ActStandardScaler
from iaml.actionables.predictors.classifier.act_catboost_classifier import ActCatBoost
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.actionables.predictors.classifier.act_linear_svc import ActLinearSVC
from iaml.actionables.predictors.classifier.act_hist_gradient_boosting_classifier import (
    ActHistGradientBoostingClassifier,
)
from iaml.actionables.predictors.classifier.act_passive_aggressive_classifier import (
    ActPassiveAggressiveClassifier,
)
from iaml.actionables.predictors.classifier.act_ridge_classifier import ActRidgeClassifier
from iaml.actionables.predictors.classifier.act_sgd_classifier import ActSGDClassifier
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.flow import Const, compile_pipeline, use
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metrics import RocAucMetric
from iaml.plots import PrecisionRecallCurvePlot, ROCAUCPlot
from tests.helpers.datasets import make_classification_data


class TestBinaryAucScores(unittest.TestCase):
    def setUp(self):
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        cache.configure(None)
        verbose = Logger().verbose
        self.addCleanup(setattr, Logger(), 'verbose', verbose)
        Logger().verbose = 0
        self.addCleanup(plt.close, 'all')
        self.X, self.y = make_classification_data(n_samples=60, seed=123)
        self.X_test, self.y_test = make_classification_data(n_samples=24, seed=124)

    def make_pipeline(self, predictor_type):
        return IAMLPipeline(
            [('scale', ActStandardScaler()), ('model', predictor_type())],
            estimator_type='classifier',
        )

    def test_margin_adapters_match_backend_auc_and_preserve_raw_inputs(self):
        for predictor_type in (ActLinearSVC, ActRidgeClassifier):
            for labels in ((0, 1), (False, True), (-3, 5), ('healthy', 'ill')):
                with self.subTest(model=predictor_type.__name__, labels=labels):
                    y = np.asarray(labels)[self.y]
                    y_test = np.asarray(labels)[self.y_test]
                    pipeline = self.make_pipeline(predictor_type)
                    self.assertFalse(hasattr(pipeline, 'decision_function'))
                    pipeline.fit(self.X, y, metrics=[RocAucMetric()])
                    transformed = pipeline.transform(self.X_test)
                    step = pipeline.predictor[1]
                    expected = step.model.decision_function(transformed[step.columns])
                    frame = self.X_test.copy(deep=True)
                    for _ in range(2):
                        np.testing.assert_allclose(pipeline.decision_function(frame), expected)
                        pd.testing.assert_frame_equal(frame, self.X_test)
                    np.testing.assert_allclose(
                        pipeline.decision_function(transformed, model_only=True), expected,
                    )
                    extra = transformed.assign(unused='text')
                    np.testing.assert_allclose(step.decision_function(extra), expected)
                    np.testing.assert_allclose(
                        pipeline.decision_function(extra, model_only=True), expected,
                    )
                    candidate = Candidate(Dataset(self.X, y), metrics=[RocAucMetric()],
                                          main_metric='ROC AUC', iaml_pipeline=pipeline)
                    self.assertAlmostEqual(
                        candidate.evaluate(frame, y_test)['ROC AUC'],
                        roc_auc_score(y_test, expected),
                    )
                    self.assertFalse(hasattr(step, 'predict_proba'))
                    self.assertFalse(hasattr(pipeline, 'predict_proba'))
                    with self.assertRaises(AttributeError):
                        candidate.predict_proba(frame)

    def test_probability_only_models_remain_valid(self):
        pipeline = self.make_pipeline(ActDecisionTreeClassifier)
        pipeline.fit(self.X, self.y)
        self.assertFalse(hasattr(pipeline, 'decision_function'))
        expected = roc_auc_score(self.y_test, pipeline.predict_proba(self.X_test)[:, 1])
        candidate = Candidate(Dataset(self.X, self.y), metrics=[RocAucMetric()],
                              iaml_pipeline=pipeline)
        self.assertAlmostEqual(candidate.evaluate(self.X_test, self.y_test)['ROC AUC'], expected)
        self.assertEqual(candidate.evaluation_report[0]['status'], 'success')

    def test_catboost_keeps_original_labels_for_auc_and_curves(self):
        labels = np.array(['healthy', 'ill'])
        step = compile_pipeline(use(ActCatBoost, iterations=Const(5), depth=Const(2)))
        pipeline = IAMLPipeline([('model', step)], estimator_type='classifier')
        pipeline.fit(self.X, labels[self.y], metrics=[RocAucMetric()])
        np.testing.assert_array_equal(pipeline.classes_, labels)
        scores = pipeline.predict_proba(self.X_test)[:, 1]
        targets = labels[self.y_test]
        expected_auc = roc_auc_score(targets, scores)
        candidate = Candidate(Dataset(self.X, labels[self.y]), metrics=[RocAucMetric()],
                              iaml_pipeline=pipeline)
        self.assertAlmostEqual(candidate.evaluate(self.X_test, targets)['ROC AUC'], expected_auc)
        expected_ap = average_precision_score(targets == labels[1], scores)
        for plot_type, expected in ((ROCAUCPlot, expected_auc),
                                    (PrecisionRecallCurvePlot, expected_ap)):
            with patch.object(plt, 'plot', wraps=plt.plot) as plotted:
                result = plot_type().compute(pipeline, self.X_test, targets)
            self.assertTrue(result.image.startswith(b'\x89PNG'))
            curve = next(call for call in plotted.call_args_list
                         if call.kwargs.get('label', '').startswith(
                             ('ROC curve', 'Precision-Recall curve')))
            self.assertIn(f'{expected:.2f}', curve.kwargs['label'])

    def test_other_margin_adapters_keep_their_fitted_feature_selection(self):
        frame = self.X.assign(unused='text')
        test_frame = self.X_test.assign(unused='text')
        for predictor_type in (ActSGDClassifier, ActPassiveAggressiveClassifier,
                               ActHistGradientBoostingClassifier):
            with self.subTest(model=predictor_type.__name__):
                step = predictor_type()
                if predictor_type is ActHistGradientBoostingClassifier:
                    step.configure({'max_iter': 10})
                step.fit(Dataset(frame, self.y))
                expected = step.model.decision_function(self.X_test[step.columns])
                np.testing.assert_allclose(step.decision_function(test_frame), expected)
                pipeline = IAMLPipeline([('model', step)], estimator_type='classifier')
                candidate = Candidate(Dataset(frame, self.y), metrics=[RocAucMetric()],
                                      iaml_pipeline=pipeline)
                self.assertAlmostEqual(
                    candidate.evaluate(test_frame, self.y_test)['ROC AUC'],
                    roc_auc_score(self.y_test, expected),
                )

    def test_auc_metric_accepts_margins_and_probabilities_in_fitted_class_order(self):
        targets = ['ill', 'healthy', 'ill', 'healthy']
        margins = np.array([2.0, -2.0, 0.1, 0.3])
        probabilities = np.array([[0.1, 0.9], [0.9, 0.1], [0.45, 0.55], [0.4, 0.6]])
        metric = RocAucMetric()
        self.assertAlmostEqual(metric.compute(targets, margins), 0.75)
        self.assertAlmostEqual(metric.compute(targets, probabilities), 0.75)
        self.assertAlmostEqual(
            metric.compute(targets, -margins, classes=['ill', 'healthy']), 0.75,
        )
        with self.assertRaisesRegex(ValueError, 'decision scores'):
            metric.compute(targets, np.zeros((4, 3)))

    def test_roc_and_precision_recall_curves_use_margin_scores(self):
        for predictor_type in (ActLinearSVC, ActRidgeClassifier):
            with self.subTest(model=predictor_type.__name__):
                labels = np.array(['healthy', 'ill'])
                pipeline = self.make_pipeline(predictor_type)
                pipeline.fit(self.X, labels[self.y])
                scores = pipeline.decision_function(self.X_test)
                targets = labels[self.y_test]
                expected_auc = roc_auc_score(targets, scores)
                expected_ap = average_precision_score(targets == pipeline.classes_[1], scores)
                for plot_type, expected in ((ROCAUCPlot, expected_auc),
                                            (PrecisionRecallCurvePlot, expected_ap)):
                    with patch.object(plt, 'plot', wraps=plt.plot) as plotted:
                        result = plot_type().compute(pipeline, self.X_test, targets)
                    self.assertTrue(result.image.startswith(b'\x89PNG'))
                    curve = next(call for call in plotted.call_args_list
                                 if call.kwargs.get('label', '').startswith(
                                     ('ROC curve', 'Precision-Recall curve')))
                    self.assertIn(f'{expected:.2f}', curve.kwargs['label'])


if __name__ == '__main__':
    unittest.main()
