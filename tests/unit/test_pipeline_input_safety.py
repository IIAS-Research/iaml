"""Inference must keep raw inputs reusable across prediction methods."""
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from iaml.actionables.normalize.act_minmax_scaler import ActMinMaxScaler
from iaml.actionables.normalize.act_standard_scaler import ActStandardScaler
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.actionables.predictors.classifier.act_logistic_regression import (
    ActLogisticRegression,
)
from iaml.actionables.predictors.survival.act_survival_tree import ActSurvivalTree
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.metrics import AccuracyMetric, RocAucMetric
from iaml.shared_cache import CacheService
from iaml.splitters import kfold_splitter


class TestPipelineInputSafety(unittest.TestCase):
    """Exercise adapters that intentionally transform DataFrames in place."""

    def setUp(self):
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        backend = CacheService()
        backend.__set_backend__({}, [], SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)
        self.X_train = pd.DataFrame({
            "age": np.linspace(31.0, 93.0, 32),
            "measurement": np.linspace(101.0, 287.0, 32),
        })
        self.y_train = np.array([0] * 16 + [1] * 16)
        self.X_test = pd.DataFrame({
            "age": [34.0, 44.0, 54.0, 69.0, 79.0, 89.0],
            "measurement": [110.0, 140.0, 170.0, 215.0, 245.0, 275.0],
        }, index=[41, 43, 47, 53, 59, 61])
        self.y_test = np.array([0, 0, 0, 1, 1, 1])

    def make_pipeline(self, scaler_type=ActStandardScaler,
                      predictor_type=ActLogisticRegression):
        scaler, predictor = scaler_type(), predictor_type()
        self.addCleanup(scaler.reset_cache)
        self.addCleanup(predictor.reset_cache)
        return IAMLPipeline(
            [("scale", scaler), ("model", predictor)],
            estimator_type="survival" if predictor_type is ActSurvivalTree else "classifier",
        )

    def reference_transform(self, pipeline, X):
        scaler = pipeline.transformers[0][1]
        return pd.DataFrame(
            scaler.scaler.transform(X[scaler.columns]),
            columns=scaler.columns, index=X.index,
        )

    def test_transform_and_fit_preserve_the_raw_frames(self):
        for scaler_type in (ActStandardScaler, ActMinMaxScaler):
            with self.subTest(scaler=scaler_type.__name__):
                pipeline = self.make_pipeline(scaler_type)
                training_before = self.X_train.copy(deep=True)
                pipeline.fit(self.X_train, self.y_train)
                pd.testing.assert_frame_equal(self.X_train, training_before)
                before = self.X_test.copy(deep=True)
                expected = self.reference_transform(pipeline, self.X_test)
                for _ in range(2):
                    transformed = pipeline.transform(self.X_test)
                    pd.testing.assert_frame_equal(transformed, expected)
                    pd.testing.assert_frame_equal(self.X_test, before)
                    self.assertIsNot(transformed, self.X_test)

    def test_predictions_are_repeatable_in_both_call_orders(self):
        for scaler_type in (ActStandardScaler, ActMinMaxScaler):
            for predictor_type in (ActLogisticRegression, ActDecisionTreeClassifier):
                with self.subTest(scaler=scaler_type.__name__, model=predictor_type.__name__):
                    pipeline = self.make_pipeline(scaler_type, predictor_type)
                    pipeline.fit(self.X_train, self.y_train)
                    transformed = self.reference_transform(pipeline, self.X_test)
                    model = pipeline.predictor[1]
                    expected = {"predict": model.predict(transformed),
                                "predict_proba": model.predict_proba(transformed)}
                    before = self.X_test.copy(deep=True)
                    for methods in (("predict", "predict_proba"),
                                    ("predict_proba", "predict")):
                        for _ in range(2):
                            for method in methods:
                                result = getattr(pipeline, method)(self.X_test)
                                np.testing.assert_allclose(result, expected[method])
                                pd.testing.assert_frame_equal(self.X_test, before)

    def test_evaluation_shares_raw_values_between_prediction_modes(self):
        for scaler_type in (ActStandardScaler, ActMinMaxScaler):
            for predictor_type in (ActLogisticRegression, ActDecisionTreeClassifier):
                with self.subTest(scaler=scaler_type.__name__, model=predictor_type.__name__):
                    pipeline = self.make_pipeline(scaler_type, predictor_type)
                    pipeline.fit(self.X_train, self.y_train)
                    transformed = self.reference_transform(pipeline, self.X_test)
                    model = pipeline.predictor[1]
                    expected = {
                        "accuracy": accuracy_score(self.y_test, model.predict(transformed)),
                        "ROC AUC": roc_auc_score(self.y_test, model.predict_proba(transformed)[:, 1]),
                    }
                    before = self.X_test.copy(deep=True)
                    for metric_types in ((AccuracyMetric, RocAucMetric),
                                         (RocAucMetric, AccuracyMetric)):
                        candidate = Candidate(
                            Dataset(self.X_train, self.y_train),
                            metrics=[metric_type() for metric_type in metric_types],
                            main_metric="accuracy", iaml_pipeline=pipeline,
                        )
                        for _ in range(2):
                            scores = candidate.evaluate(self.X_test, self.y_test)
                            self.assertEqual(set(scores), set(expected))
                            for key in expected:
                                self.assertAlmostEqual(scores[key], expected[key])
                            pd.testing.assert_frame_equal(self.X_test, before)

    def test_external_evaluation_preserves_cross_validation_results(self):
        dataset = Dataset(self.X_train, self.y_train)
        candidate = Candidate(
            dataset, metrics=[AccuracyMetric(), RocAucMetric()], main_metric="accuracy",
            iaml_pipeline=self.make_pipeline(),
        )
        scores = candidate.training_evaluate(
            dataset, splitter=kfold_splitter, cache_split=False, store_audit=True,
        )
        self.assertEqual(set(scores), {"accuracy", "ROC AUC"})
        before_scores = deepcopy(candidate.computed_metrics)
        before_folds = deepcopy(candidate.fold_metrics)
        before_coverage = deepcopy(candidate.metric_coverage)
        before_report = deepcopy(candidate.metric_report)
        before_audit = deepcopy(candidate.training_audit)
        candidate.pipeline.fit(self.X_train, self.y_train)
        before = self.X_test.copy(deep=True)
        for _ in range(2):
            candidate.evaluate(self.X_test, self.y_test)
            candidate.predict(self.X_test)
            candidate.predict_proba(self.X_test)
            pd.testing.assert_frame_equal(self.X_test, before)
            self.assertEqual(candidate.computed_metrics, before_scores)
            self.assertEqual(candidate.fold_metrics, before_folds)
            self.assertEqual(candidate.metric_coverage, before_coverage)
            self.assertEqual(candidate.metric_report, before_report)
            self.assertEqual(candidate.training_audit, before_audit)

    def test_survival_predictions_do_not_mutate_the_raw_frame(self):
        pipeline = self.make_pipeline(predictor_type=ActSurvivalTree)
        y = [(index % 4 != 0, float(index + 5)) for index in range(len(self.X_train))]
        pipeline.fit(self.X_train, y)
        transformed = self.reference_transform(pipeline, self.X_test)
        model = pipeline.predictor[1]
        expected_risk = model.predict(transformed)
        expected_survival = model.predict_survival_function(transformed)
        before = self.X_test.copy(deep=True)
        for _ in range(2):
            np.testing.assert_allclose(pipeline.predict(self.X_test), expected_risk)
            actual_survival = pipeline.predict_survival_function(self.X_test)
            for actual, expected in zip(actual_survival, expected_survival):
                np.testing.assert_array_equal(actual.x, expected.x)
                np.testing.assert_allclose(actual.y, expected.y)
            pd.testing.assert_frame_equal(self.X_test, before)


if __name__ == "__main__":
    unittest.main()
