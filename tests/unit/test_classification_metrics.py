"""Regression tests for the positive class used by classification metrics."""
from itertools import permutations
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.metrics import F1ScoreMetric, PrecisionMetric, RecallMetric


METRICS = (
    (PrecisionMetric, precision_score, 2 / 3),
    (RecallMetric, recall_score, 1.0),
    (F1ScoreMetric, f1_score, 0.8),
)


class TestClassificationMetrics(unittest.TestCase):
    """Scores must identify a fixed class, independently of row ordering."""

    def test_row_order_does_not_change_scores(self):
        for negative, positive in (
            (0, 1), (-1, 1), (False, True), (1, 2), (2, 3),
            ("healthy", "ill"),
        ):
            y = np.array([negative, negative, positive, positive])
            prediction = np.array([negative, positive, positive, positive])
            for metric_class, _, expected in METRICS:
                metric = metric_class()
                for order in permutations(range(len(y))):
                    indices = list(order)
                    with self.subTest(metric=metric_class.__name__, labels=(negative, positive),
                                      order=order):
                        self.assertAlmostEqual(
                            metric.compute(y[indices], prediction[indices]), expected
                        )

    def test_lists_and_pandas_targets_with_non_default_index(self):
        index = [10, 30, 20, 40]
        for convert in (
            list,
            lambda values: pd.Series(values, index=index),
            lambda values: pd.DataFrame({"target": values}, index=index),
        ):
            for metric_class, _, expected in METRICS:
                with self.subTest(metric=metric_class.__name__, container=convert):
                    self.assertAlmostEqual(
                        metric_class().compute(convert([0, 0, 1, 1]), convert([0, 1, 1, 1])),
                        expected,
                    )

    def test_explicit_positive_class(self):
        for negative, positive in ((0, 1), ("case", "control")):
            y = np.array([negative, negative, positive, positive])
            prediction = np.array([negative, positive, positive, positive])
            for metric_class, score, _ in METRICS:
                expected = score(y, prediction, pos_label=negative, zero_division=0)
                metric = metric_class(pos_label=negative)
                for order in ([0, 1, 2, 3], [2, 1, 0, 3]):
                    with self.subTest(metric=metric_class.__name__, positive=negative, order=order):
                        self.assertAlmostEqual(
                            metric.compute(y[order], prediction[order]), expected
                        )

    def test_all_negative_standard_labels_score_zero(self):
        for negative in (0, -1, False):
            for metric_class, _, _ in METRICS:
                with self.subTest(metric=metric_class.__name__, negative=negative):
                    self.assertEqual(metric_class().compute([negative] * 4, [negative] * 4), 0)

    def test_training_labels_define_positive_class_for_single_class_test_set(self):
        for metric_class, _, _ in METRICS:
            for y_train in (["healthy", "ill"], ["ill", "healthy"]):
                with self.subTest(metric=metric_class.__name__, y_train=y_train):
                    self.assertEqual(
                        metric_class().compute(["healthy"] * 4, ["healthy"] * 4,
                                               y_train=y_train),
                        0,
                    )

    def test_explicit_class_can_be_absent_from_test_set(self):
        for metric_class, _, _ in METRICS:
            with self.subTest(metric=metric_class.__name__):
                self.assertEqual(
                    metric_class(pos_label="ill").compute(["healthy"] * 4, ["healthy"] * 4),
                    0,
                )

    def test_ambiguous_single_class_requires_explicit_label(self):
        for metric_class, _, _ in METRICS:
            for kwargs in ({}, {"y_train": ["healthy"] * 4}):
                with self.subTest(metric=metric_class.__name__, kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, "pos_label"):
                        metric_class().compute(["healthy"] * 4, ["healthy"] * 4, **kwargs)

    def test_invalid_explicit_label_is_rejected(self):
        for metric_class, _, _ in METRICS:
            with self.subTest(metric=metric_class.__name__):
                with self.assertRaises(ValueError):
                    metric_class(pos_label="unknown").compute(["healthy", "ill"],
                                                               ["healthy", "ill"])

    def test_multiclass_weighted_scores_are_preserved(self):
        for repetitions in (1, 6):
            y = np.tile([0, 1, 2], repetitions)
            prediction = np.tile([0, 2, 2], repetitions)
            for metric_class, score, _ in METRICS:
                with self.subTest(metric=metric_class.__name__, repetitions=repetitions):
                    self.assertAlmostEqual(
                        metric_class().compute(y, prediction),
                        score(y, prediction, average="weighted", zero_division=0),
                    )

    def test_multiclass_reference_with_missing_test_classes(self):
        for y_train in ([0, 1, 2], np.tile([0, 1, 2], 6)):
            for y, prediction in (
                ([0, 0, 1, 1], [0, 1, 1, 1]),
                ([0, 0], [0, 0]),
            ):
                for metric_class, score, _ in METRICS:
                    with self.subTest(metric=metric_class.__name__, y_train=y_train, y=y):
                        self.assertAlmostEqual(
                            metric_class().compute(y, prediction, y_train=y_train),
                            score(y, prediction, average="weighted", zero_division=0),
                        )

    def test_multilabel_sample_scores_are_preserved(self):
        for n_labels in (2, 3):
            y = np.tile([[1, 0, 1], [0, 1, 0]], (6, 1))[:, :n_labels]
            prediction = np.tile([[1, 0, 0], [0, 1, 1]], (6, 1))[:, :n_labels]
            for metric_class, score, _ in METRICS:
                with self.subTest(metric=metric_class.__name__, n_labels=n_labels):
                    self.assertAlmostEqual(
                        metric_class().compute(y, prediction),
                        score(y, prediction, average="samples", zero_division=0),
                    )

    def test_candidate_evaluation_uses_training_labels(self):
        train = pd.DataFrame({"x": range(12)})
        candidate = Candidate(Dataset(train, ["healthy", "ill"] * 6))
        candidate.metrics = [metric_class() for metric_class, _, _ in METRICS]
        candidate.pipeline = Mock(have_model=True)
        candidate.pipeline.predict.return_value = np.array(["healthy"] * 4)

        scores = candidate.evaluate(pd.DataFrame({"x": range(4)}), ["healthy"] * 4)

        self.assertEqual(scores, {"precision": 0.0, "recall": 0.0, "f1_score": 0.0})
