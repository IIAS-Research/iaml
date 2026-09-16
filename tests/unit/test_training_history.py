import sys
import unittest

sys.path.append('./src')

from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.metrics.accuracy_metric import AccuracyMetric
from iaml.splitters.kfold_splitter import kfold_splitter
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)

from tests.helpers.datasets import make_classification_data


def two_fold_splitter(dataset):
    return kfold_splitter(dataset, nb_folds=2)


class TestTrainingHistory(unittest.TestCase):

    def test_candidate_training_audit_captures_fold_metrics(self):
        X, y = make_classification_data(n_samples=24, seed=101)
        dataset = Dataset(X, y)
        metric = AccuracyMetric()
        candidate = Candidate(dataset, metrics=[metric], main_metric=metric)
        candidate = ActDecisionTreeClassifier().run(candidate)[0]

        metrics = candidate.training_evaluate(
            dataset,
            splitter=two_fold_splitter,
            cache_split=False,
            store_audit=True,
        )

        self.assertIn("accuracy", metrics)
        self.assertIsNotNone(candidate.training_audit)
        self.assertEqual(candidate.training_audit["status"], "success")
        self.assertEqual(len(candidate.training_audit["fold_metrics"]), 2)
        self.assertEqual(candidate.training_audit["pipeline"]["steps"][0]["role"], "predictor")
        self.assertEqual(
            candidate.training_audit["pipeline"]["steps"][0]["class"],
            "ActDecisionTreeClassifier",
        )
