"""Declarative analytics preserve variants, snapshots and partial failures."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import ActDecisionTreeClassifier
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metric import Metric
from iaml.metrics import AccuracyMetric, PrecisionMetric
from iaml.splitters import kfold_splitter
from iaml.statistic import Statistic
from iaml.statistics import MeanStatistic, BoundStatistic
from iaml.study_analyses import analyses_signature, compile_analyses, compile_metrics


def collection(kind, *entries):
    """Supply the compiler protocol without involving pipeline execution."""
    specs = [SimpleNamespace(component=type(component), alias=alias,
                             instantiate=lambda component=component: deepcopy(component))
             for component, alias in entries]
    return SimpleNamespace(kind=kind, resolved=lambda: specs)


class SometimesMissingMetric(Metric):
    def __init__(self):
        self.calls = 0

    def __str__(self):
        return "sometimes_missing"

    def compute(self, y, y_pred, **kwargs):
        self.calls += 1
        if self.calls % 2 == 0:
            raise ValueError("Unavailable on this fold")
        return 0.8


class BrokenStatistic(Statistic):
    def __str__(self):
        return "broken_statistic"

    def suitable(self, dataset):
        return True

    def compute(self, dataset, **kwargs):
        raise ValueError("Independent failure")


class BrokenExplanation:
    def suitable(self, dataset):
        return True

    def compute(self, pipeline, X, y=None, **kwargs):
        raise ValueError("Independent failure")


class TestFlowAnalytics(unittest.TestCase):
    def setUp(self):
        verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", verbose)
        self.X = pd.DataFrame({"feature": np.arange(24, dtype=float) + 0.5})
        self.y = np.tile([0, 1], 12)
        self.dataset = Dataset(self.X, self.y)


    def test_legacy_objective_parameters_and_collisions(self):
        source = collection("metrics", (PrecisionMetric(), None))
        definitions, objective, _ = compile_metrics(source, PrecisionMetric(pos_label=0), self.dataset)
        self.assertEqual(objective, "precision")
        self.assertEqual(definitions[0].component.pos_label, 0)
        with self.assertRaises(ValueError):
            compile_analyses(collection("metrics", (PrecisionMetric(0), None), (PrecisionMetric(1), None)))
        with self.assertRaises(ValueError):
            compile_metrics(collection("metrics", (PrecisionMetric(0), "zero"),
                                       (PrecisionMetric(1), "one")), PrecisionMetric(), self.dataset)
        with self.assertRaises(ValueError):
            compile_metrics(source, "absent", self.dataset)








    def test_supplied_objective_is_not_reconstructed_during_compilation(self):
        from iaml.flow import metrics

        source = metrics(PrecisionMetric)
        metric = PrecisionMetric(pos_label=0)
        with patch.object(PrecisionMetric, "__init__", side_effect=AssertionError("Do not recreate the objective")):
            definitions, objective, _ = compile_metrics(source, metric, self.dataset)
        self.assertEqual(objective, "precision")
        self.assertEqual(definitions[0].component.pos_label, 0)


    def test_unsuitable_secondary_remains_excluded_for_explicit_objective(self):
        from iaml.flow import metrics, use

        definitions, _, report = compile_metrics(
            metrics(use(PrecisionMetric).named("objective"),
                    use(AccuracyMetric).named("secondary")),
            "objective", self.dataset,
        )
        self.assertEqual([definition.key for definition in definitions], ["objective"])
        self.assertEqual(report, [{"key": "secondary", "status": "inapplicable",
                                  "reason": "Not suitable for target 'binary'"}])




if __name__ == "__main__":
    unittest.main()
