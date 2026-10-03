"""SVC calibration follows every requested metric and explanation."""
import unittest
from unittest.mock import patch

import numpy as np

from iaml.actionables.predictors.classifier.act_svm_svc import ActSVMSVC
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.metric import Metric
from iaml.metrics import AccuracyMetric, RocAucMetric
from iaml.plots import PrecisionRecallCurvePlot, ROCAUCPlot
from iaml.splitters import kfold_splitter
from tests.helpers.datasets import make_classification_data


class ProbabilityMetric(Metric):
    needed_prediction = 'predict_proba'

    def __str__(self):
        return 'svc_probability_metric'

    def compute(self, y, y_pred, **kwargs):
        return float(np.mean(y_pred[:, 1]))


class UnknownExplanation:
    def suitable(self, dataset):
        raise ValueError('Output requirements are unknown')


class SVCPredictionRequirementTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_classification_data(n_samples=40, seed=19)

    @staticmethod
    def pipeline():
        return IAMLPipeline([('svc', ActSVMSVC())], estimator_type='classifier')

    def test_rank_metrics_and_plots_use_svc_without_probability_calibration(self):
        for metrics, plots in (([AccuracyMetric()], []),
                               ([RocAucMetric()], [ROCAUCPlot(), PrecisionRecallCurvePlot()])):
            with self.subTest(metrics=metrics):
                pipeline = self.pipeline()
                pipeline.fit(self.X, self.y, metrics=metrics, explanations=plots)
                self.assertFalse(pipeline.predictor[1].model.probability)
                self.assertFalse(hasattr(pipeline, 'predict_proba'))
                self.assertTrue(np.isfinite(pipeline.decision_function(self.X)).all())
                candidate = Candidate(Dataset(self.X, self.y), metrics=metrics,
                                      iaml_pipeline=pipeline)
                self.assertEqual(len(candidate.evaluate(self.X, self.y)), len(metrics))

    def test_secondary_probability_metric_and_unknown_outputs_keep_probabilities(self):
        cases = (([RocAucMetric(), ProbabilityMetric()], []),
                 ([RocAucMetric()], [UnknownExplanation()]), (None, None))
        for metrics, plots in cases:
            with self.subTest(metrics=metrics, plots=plots):
                pipeline = self.pipeline()
                pipeline.fit(self.X, self.y, metrics=metrics, explanations=plots)
                self.assertTrue(pipeline.predictor[1].model.probability)
                self.assertEqual(pipeline.predict_proba(self.X).shape, (40, 2))

    def test_cv_passes_requirements_and_refit_can_request_probabilities(self):
        candidate = Candidate(Dataset(self.X, self.y), metrics=[RocAucMetric()],
                              main_metric=RocAucMetric(), iaml_pipeline=self.pipeline())
        modes = []
        original_fit = ActSVMSVC.fit

        def fit(step, dataset):
            result = original_fit(step, dataset)
            modes.append(step.model.probability)
            return result

        with patch.object(ActSVMSVC, 'fit', fit):
            self.assertTrue(candidate.training_evaluate(
                candidate.dataset, splitter=kfold_splitter, cache_split=False))
        self.assertEqual(modes, [False] * 5)
        candidate.pipeline.fit(self.X, self.y, metrics=[RocAucMetric(), ProbabilityMetric()])
        self.assertTrue(hasattr(candidate.pipeline, 'predict_proba'))


if __name__ == '__main__':
    unittest.main()
