"""Regression tests for candidate copies and probability predictions."""
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from iaml.actionable import Actionable
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.candidate import Candidate
from iaml.data_type import DataType
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.metrics import AccuracyMetric, PrecisionMetric


class TestCandidateCopies(unittest.TestCase):
    def make_candidate(self, metric=None, estimator_type="classifier"):
        dataset = Dataset(
            pd.DataFrame({"feature": np.arange(12, dtype=float) + 0.5}),
            [0, 0, 1, 0, 1, 1] * 2,
            columns_types={"feature": DataType.NUMERIC},
        )
        return Candidate(
            dataset,
            metrics=[metric] if metric is not None else [],
            main_metric=metric,
            iaml_pipeline=IAMLPipeline(estimator_type=estimator_type),
        )

    def test_copies_preserve_the_requested_metric(self):
        for metric in (AccuracyMetric(), PrecisionMetric(pos_label=0)):
            for method in ("to_output", "to_input"):
                with self.subTest(metric=str(metric), method=method):
                    candidate = self.make_candidate(metric)
                    copied = getattr(candidate, method)()
                    self.assertEqual(copied.main_metric, str(metric))
                    predictions = [0, 1, 1, 1, 1, 1] * 2
                    self.assertEqual(
                        copied.metrics[0].compute(copied.dataset.y, predictions),
                        metric.compute(candidate.dataset.y, predictions),
                    )
                    self.assertIsNot(copied.metrics, candidate.metrics)
                    self.assertIsNot(copied.dataset, candidate.dataset)
                    self.assertIsNot(copied.pipeline, candidate.pipeline)

    def test_copies_preserve_the_default_metric_for_each_task(self):
        for estimator_type, expected in (
            ("classifier", "balanced_accuracy"),
            ("regressor", "r2_score"),
            ("survival", "concordance_index_ipcw"),
        ):
            with self.subTest(estimator_type=estimator_type):
                candidate = self.make_candidate(estimator_type=estimator_type)
                self.assertEqual(candidate.to_output().to_input().main_metric, expected)

    def test_generated_candidates_are_ranked_by_the_requested_metric(self):
        candidate = self.make_candidate(AccuracyMetric())
        cleaning, predictor = Actionable(), ActDecisionTreeClassifier()
        self.addCleanup(cleaning.reset_cache)
        self.addCleanup(predictor.reset_cache)

        cleaned = cleaning.run(candidate)[0]
        generated = predictor.run(cleaned)[0]
        self.assertIsNotNone(generated.pipeline.predictor)
        self.assertEqual(generated.main_metric, "accuracy")

        alternative = generated.to_output()
        generated.computed_metrics = {"accuracy": 0.9, "balanced_accuracy": 0.5}
        alternative.computed_metrics = {"accuracy": 0.8, "balanced_accuracy": 0.95}
        self.assertEqual(generated.get_main_metric_value(), 0.9)
        self.assertGreater(generated, alternative)


class TestCandidateProbabilities(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"feature": [10.0, 11.0, 12.0, 20.0, 21.0, 22.0]})

    def test_probabilities_include_preprocessing_and_class_order(self):
        for labels in ([2, 2, 7, 2, 7, 7], ["a", "b", "a", "b", "c", "c"]):
            with self.subTest(labels=labels):
                scaler = StandardScaler().fit(self.X)
                transformed = scaler.transform(self.X)
                model = DecisionTreeClassifier(
                    max_depth=1, min_samples_leaf=3, random_state=0
                ).fit(transformed, labels)
                candidate = Candidate(
                    Dataset(self.X, labels),
                    iaml_pipeline=IAMLPipeline(
                        [("scale", scaler), ("tree", model)],
                        estimator_type="classifier",
                    ),
                )

                probabilities = candidate.predict_proba(self.X)
                expected = model.predict_proba(transformed)
                np.testing.assert_allclose(probabilities, expected)
                np.testing.assert_allclose(probabilities.sum(axis=1), 1)
                self.assertTrue(np.any((probabilities > 0) & (probabilities < 1)))
                np.testing.assert_array_equal(
                    candidate.predict(self.X), model.classes_[probabilities.argmax(axis=1)]
                )

    def test_model_without_probabilities_raises_instead_of_returning_classes(self):
        labels = [0, 0, 0, 1, 1, 1]
        model = LinearSVC(random_state=0, max_iter=10000).fit(self.X, labels)
        candidate = Candidate(
            Dataset(self.X, labels),
            iaml_pipeline=IAMLPipeline([("svm", model)], estimator_type="classifier"),
        )

        with self.assertRaises(AttributeError):
            candidate.predict_proba(self.X)
