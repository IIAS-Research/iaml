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
from iaml.study_analyses import (
    analyses_signature, canonical_statistics, compile_analyses,
    compile_metrics, compute_statistics, copy_statistics,
)


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

    def test_metric_variants_use_aliases_and_exact_objective(self):
        source = collection("metrics", (PrecisionMetric(pos_label=0), "negative"),
                            (PrecisionMetric(pos_label=1), "positive"))
        definitions, objective, _ = compile_metrics(source, "positive", self.dataset)
        model = DummyClassifier(strategy="constant", constant=1).fit(self.X, self.y)
        candidate = Candidate(
            self.dataset, main_metric=objective, metric_definitions=definitions,
            iaml_pipeline=IAMLPipeline([("model", model)], estimator_type="classifier"),
        )
        self.assertEqual(candidate.evaluate(self.X, self.y), {"negative": 0.0, "positive": 0.5})
        candidate.computed_metrics = candidate.evaluate(self.X, self.y)
        self.assertEqual(candidate.get_main_metric_value(), 0.5)
        self.assertEqual(candidate.get_main_metric().pos_label, 1)
        self.assertIn("`negative`", candidate.describe_metrics())

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


    def test_secondary_coverage_does_not_invent_zero_or_partial_mean(self):
        definitions = compile_analyses(collection("metrics", (AccuracyMetric(), "objective"),
                                                 (SometimesMissingMetric(), "secondary")))
        candidate = Candidate(self.dataset, main_metric="objective", metric_definitions=definitions)
        candidate.pipeline.set_model(ActDecisionTreeClassifier())
        signature = candidate.evaluation_context_signature()
        scores = candidate.training_evaluate(
            self.dataset, splitter=partial(kfold_splitter, nb_folds=2), cache_split=False,
        )
        self.assertIn("objective", scores)
        self.assertNotIn("secondary", scores)
        self.assertEqual(candidate.metric_coverage["secondary"]["available"], 1)
        self.assertEqual(candidate.metric_coverage["secondary"]["total"], 2)
        self.assertEqual(candidate.metric_coverage["secondary"]["values"], [0.8])
        self.assertEqual(len(candidate.fold_metrics), 2)
        self.assertEqual(signature, candidate.evaluation_context_signature())
        self.assertEqual({record["fold"] for record in candidate.metric_report}, {1, 2})

    def test_copy_statistics_does_not_share_nested_cells(self):
        table = pd.DataFrame({"feature": [{"counts": [1, 2, 3]}]})
        table.attrs["metadata"] = {"values": ["original"]}
        copied = copy_statistics(table)
        copied.iloc[0, 0]["counts"][0] = 99
        copied.attrs["metadata"]["values"].append("edited")
        self.assertEqual(table.iloc[0, 0], {"counts": [1, 2, 3]})
        self.assertEqual(table.attrs["metadata"], {"values": ["original"]})

    def test_statistics_aliases_keep_canonical_plot_rows_and_partial_results(self):
        definitions = compile_analyses(collection("statistics", (MeanStatistic(), "average"),
                                                 (BoundStatistic(), "extremes"),
                                                 (BrokenStatistic(), "failed")))
        table, report = compute_statistics(definitions, self.dataset)
        self.assertIn("average", table.index)
        self.assertIn("extremes : min", table.index)
        self.assertIn("extremes : max", table.index)
        canonical = canonical_statistics(table)
        self.assertIn("mean", canonical.index)
        self.assertIn("min", canonical.index)
        self.assertEqual(table.attrs["iaml_dataset_fingerprint"], self.dataset.fingerprint())
        self.assertEqual(table.attrs["iaml_statistics_signature"], analyses_signature(definitions))
        self.assertEqual(next(r for r in report if r["key"] == "failed")["status"], "error")
        canonical.iloc[0, 0] = -999
        self.assertNotEqual(table.iloc[0, 0], -999)




    def test_supplied_objective_is_not_reconstructed_during_compilation(self):
        from iaml.flow import metrics

        source = metrics(PrecisionMetric)
        metric = PrecisionMetric(pos_label=0)
        with patch.object(PrecisionMetric, "__init__", side_effect=AssertionError("Do not recreate the objective")):
            definitions, objective, _ = compile_metrics(source, metric, self.dataset)
        self.assertEqual(objective, "precision")
        self.assertEqual(definitions[0].component.pos_label, 0)

    def test_explicit_objective_keeps_accuracy_despite_automatic_balancing_filter(self):
        from iaml.flow import metrics, use

        self.assertFalse(AccuracyMetric().suitable(self.X, self.y, self.dataset.type_of_target))
        for objective in (AccuracyMetric(), "requested_accuracy"):
            with self.subTest(objective=objective):
                definitions, key, report = compile_metrics(
                    metrics(use(AccuracyMetric).named("requested_accuracy")),
                    objective, self.dataset,
                )
                self.assertEqual(key, "requested_accuracy")
                self.assertEqual([definition.key for definition in definitions], [key])
                self.assertEqual(report[0]["status"], "selected")
                candidate = Candidate(self.dataset, main_metric=key,
                                      metric_definitions=definitions)
                candidate.pipeline.set_model(ActDecisionTreeClassifier())
                scores = candidate.training_evaluate(
                    self.dataset, splitter=partial(kfold_splitter, nb_folds=2), cache_split=False,
                )
                self.assertTrue(np.isfinite(scores[key]))

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

    def test_candidate_audit_preserves_variant_and_retained_search_policy(self):
        from iaml.flow import Const, Int, use

        recipe = use(ActDecisionTreeClassifier, max_depth=Int(2, 8, initial=4)).named("tree")
        recipe.configure(max_depth=Const(10))
        model = recipe.instantiate()
        alternative = use(ActDecisionTreeClassifier, max_depth=Int(1, 3, initial=2)).named("short_tree").instantiate()
        model._flow_alternatives = (model, alternative)
        model._flow_choice_id = "models"
        candidate = Candidate(self.dataset)
        candidate.pipeline.set_model(model)
        copied = candidate.to_output()
        summary = copied.pipeline_audit_summary()
        step = summary["steps"][0]
        self.assertEqual(step["alias"], "tree")
        self.assertEqual(step["node_id"], recipe.node_id)
        self.assertEqual(step["variant_id"], recipe.node_id)
        self.assertEqual(step["choice_id"], "models")
        self.assertTrue(step["search_policy"]["parameters"]["max_depth"]["fixed"])
        self.assertEqual(step["search_policy"]["parameters"]["max_depth"]["domain"], [2, 8])
        self.assertEqual([item["alias"] for item in step["search_policy"]["alternatives"]],
                         ["tree", "short_tree"])
        json.dumps(summary)

    def test_iaml_descriptive_before_fit_caches_configuration_and_copies_object_cells(self):
        from iaml import IAML, TopKValueCountsStatistic
        from iaml.flow import statistics, use

        frame = pd.DataFrame({"category": ["a", "b", "c", "a"] * 6})
        search = IAML(statistics=statistics(use(TopKValueCountsStatistic, k=2).named("categories")))
        original_compute = TopKValueCountsStatistic.compute
        with patch.object(TopKValueCountsStatistic, "compute", autospec=True,
                          side_effect=original_compute) as compute:
            first = search.get_descriptive_statistics(frame, self.y)
            self.assertIsNone(search._last_dataset)
            self.assertEqual(len(first.iloc[0, 0]), 2)
            first.iloc[0, 0].append(("unwanted", 99, 0.5))
            second = search.get_descriptive_statistics(frame, self.y)
            self.assertEqual(len(second.iloc[0, 0]), 2)
            self.assertEqual(compute.call_count, 1)
            search.statistics["categories"].configure(k=1)
            third = search.get_descriptive_statistics(frame, self.y)
            self.assertEqual(len(third.iloc[0, 0]), 1)
            self.assertEqual(compute.call_count, 2)


if __name__ == "__main__":
    unittest.main()
