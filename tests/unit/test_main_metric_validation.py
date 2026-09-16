"""Reject candidates whose main metric cannot be evaluated reliably."""
import unittest
from unittest.mock import patch

import numpy as np

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.actionables.predictors.classifier.act_linear_svc import ActLinearSVC
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.logger import Logger
from iaml.metrics import AccuracyMetric, RocAucMetric
from iaml.splitters import kfold_splitter
from tests.helpers.datasets import make_classification_data


class TestMainMetricValidation(unittest.TestCase):
    def setUp(self):
        previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", previous_verbose)
        self.dataset = Dataset(*make_classification_data(n_samples=40, seed=93))

    def make_candidate(self, main_metric, model=None):
        candidate = Candidate(
            self.dataset, metrics=[main_metric], main_metric=main_metric,
        )
        candidate.pipeline.set_model(model if model is not None else ActDecisionTreeClassifier())
        return candidate

    def evaluate(self, candidate, splitter=kfold_splitter):
        return candidate.training_evaluate(
            self.dataset, splitter=splitter, cache_split=False, store_audit=True,
        )

    def assert_failed(self, candidate):
        self.assertEqual(candidate.computed_metrics, {})
        self.assertEqual(candidate.get_main_metric_score(), float("-inf"))
        self.assertEqual(candidate.training_audit["status"], "failed")
        self.assertEqual(candidate.training_audit["metrics"], {})

    def test_linear_svc_is_rejected_when_auc_is_the_main_metric(self):
        candidate = self.make_candidate(RocAucMetric(), ActLinearSVC())
        candidate.add_metric(AccuracyMetric())
        self.assertEqual(self.evaluate(candidate), {})
        self.assert_failed(candidate)
        self.assertIn(candidate.main_metric, candidate.training_audit["error"])

    def test_failed_secondary_metric_does_not_reject_valid_main_metric(self):
        candidate = self.make_candidate(AccuracyMetric(), ActLinearSVC())
        candidate.add_metric(RocAucMetric())
        scores = self.evaluate(candidate)
        self.assertTrue(np.isfinite(scores[candidate.main_metric]))
        self.assertEqual(candidate.training_audit["status"], "success")

    def test_invalid_fold_discards_partial_and_previous_scores(self):
        for invalid in (None, np.nan, np.inf, -np.inf, np.array([0.2, 0.8]), ValueError("failed")):
            with self.subTest(invalid=invalid):
                metric = AccuracyMetric()
                candidate = self.make_candidate(metric)
                self.assertTrue(self.evaluate(candidate))
                with patch.object(metric, "compute", side_effect=[0.8, invalid]):
                    self.assertEqual(self.evaluate(candidate), {})
                self.assert_failed(candidate)
                self.assertEqual(len(candidate.training_audit["fold_metrics"]), 1)
                self.assertIn("fold 2", candidate.training_audit["error"])

    def test_zero_is_a_valid_main_metric(self):
        metric = AccuracyMetric()
        candidate = self.make_candidate(metric)
        with patch.object(metric, "compute", return_value=0.0):
            self.assertEqual(self.evaluate(candidate)[candidate.main_metric], 0.0)
        self.assertEqual(candidate.get_main_metric_score(), 0.0)
        self.assertEqual(candidate.training_audit["status"], "success")

    def test_no_folds_does_not_produce_a_zero_score(self):
        candidate = self.make_candidate(AccuracyMetric())
        self.assertEqual(self.evaluate(candidate, splitter=lambda dataset: iter(())), {})
        self.assert_failed(candidate)
