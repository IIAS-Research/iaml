"""Evaluation caches must follow labels, groups, and evaluation settings."""
from functools import partial
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.logger import Logger
from iaml.metric import Metric
from iaml.metrics.precision_metric import PrecisionMetric
from iaml.shared_cache import CacheService
from iaml.splitters import kfold_splitter


class TargetMeanMetric(Metric):
    """Expose the labels received by the scoring stage."""

    def __str__(self):
        return "target_mean"

    def compute(self, y, y_pred, **kwargs):
        return float(np.mean(y))


class SynchronousExecutor:
    """Run the real evaluation worker without starting processes."""

    def __init__(self):
        self.submissions = 0
        self.pending = []
        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def submit(self, function, *args, deadline=None, **kwargs):
        self.submissions += 1
        self.pending.append(function(*args, **kwargs))
        self.callback()
        return True

    def join(self, timeout, *, cancel_pending=None):
        result, self.pending = self.pending, []
        return result


class StatefulSplitter:
    """Increase the fold count each time this callable is executed."""

    def __init__(self):
        self.folds = 2

    def __call__(self, dataset):
        folds = self.folds
        self.folds += 1
        return kfold_splitter(dataset, nb_folds=folds)


class TestEvaluationCache(unittest.TestCase):
    """Use the production cache service with an in-process backend."""

    def setUp(self):
        self.previous_backend = Cache()._backend
        self.saved = {}
        backend = CacheService()
        backend.__set_backend__(self.saved, [], SimpleNamespace(value=False), RLock(), 100)
        Cache().configure(backend)
        self.addCleanup(Cache().configure, self.previous_backend)
        self.previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", self.previous_verbose)
        self.X = pd.DataFrame({"feature": np.arange(24, dtype=float)})
        self.y = np.tile([0, 1], 12)
        self.splitter = partial(kfold_splitter, nb_folds=2)

    def make_candidate(self, dataset, metric=None):
        metric = metric if metric is not None else TargetMeanMetric()
        candidate = Candidate(dataset, metrics=[metric], main_metric=metric)
        candidate.pipeline.set_model(ActDecisionTreeClassifier())
        return candidate

    def make_iaml(self):
        # Only initialize the state used by the evaluation stage under test.
        engine = IAML.__new__(IAML)
        engine.executor = SynchronousExecutor()
        engine.splitter = self.splitter
        engine.max_stage_duration = 60
        engine.keep_training_history = True
        engine.training_history = []
        engine._training_history_seen = set()
        return engine

    def evaluate(self, engine, dataset, metric=None):
        result = engine._IAML__run_evaluations(
            [self.make_candidate(dataset, metric)], dataset, timeout=60,
        )
        self.assertEqual(len(result), 1)
        return result[0]

    def split_entries(self):
        return [value for (key, _), value in self.saved.items() if key.startswith("splits_")]

    def test_split_cache_keeps_current_targets_and_reuses_identical_inputs(self):
        dataset = Dataset(self.X, self.y)
        candidate = self.make_candidate(dataset)
        first = candidate.training_evaluate(dataset, splitter=self.splitter)
        self.assertEqual(first["target_mean"], 0.5)
        self.assertEqual(candidate.training_evaluate(dataset, splitter=self.splitter), first)
        self.assertEqual(len(self.split_entries()), 1)

        changed = Dataset(self.X, np.ones(24, dtype=int))
        actual = candidate.training_evaluate(changed, splitter=self.splitter)
        self.assertEqual(actual["target_mean"], 1.0)
        self.assertEqual(len(self.split_entries()), 2)

    def test_split_cache_keeps_current_groups(self):
        groups = pd.DataFrame({"group": np.repeat(np.arange(12), 2)})
        dataset = Dataset(self.X, self.y, groups=groups)
        candidate = self.make_candidate(dataset)
        candidate.training_evaluate(dataset, splitter=self.splitter)
        changed = Dataset(self.X, self.y, groups=groups + 100)
        candidate.training_evaluate(changed, splitter=self.splitter)

        entries = self.split_entries()
        self.assertEqual(len(entries), 2)
        self.assertTrue(all(train.groups["group"].min() >= 100 for train, _ in entries[1]))

    def test_split_cache_distinguishes_partial_splitter_arguments(self):
        dataset = Dataset(self.X, self.y)
        candidate = self.make_candidate(dataset)
        candidate.training_evaluate(dataset, splitter=self.splitter, store_audit=True)
        self.assertEqual(len(candidate.fold_metrics), 2)
        candidate.training_evaluate(
            dataset, splitter=partial(kfold_splitter, nb_folds=3), store_audit=True,
        )
        self.assertEqual(len(candidate.fold_metrics), 3)
        self.assertEqual(len(self.split_entries()), 2)

    def test_unserializable_splitter_bypasses_split_cache(self):
        dataset = Dataset(self.X, self.y)
        candidate = self.make_candidate(dataset)
        splitter = lambda current: kfold_splitter(current, nb_folds=2)
        for _ in range(2):
            result = candidate.training_evaluate(dataset, splitter=splitter)
            self.assertEqual(result["target_mean"], 0.5)
        self.assertEqual(self.split_entries(), [])

    def test_score_cache_keeps_current_targets_and_reuses_identical_inputs(self):
        engine = self.make_iaml()
        dataset = Dataset(self.X, self.y)
        first = self.evaluate(engine, dataset)
        second = self.evaluate(engine, dataset)
        self.assertEqual(first.computed_metrics, second.computed_metrics)
        self.assertEqual(engine.executor.submissions, 1)

        changed = Dataset(self.X, np.ones(24, dtype=int))
        result = self.evaluate(engine, changed)
        self.assertEqual(result.computed_metrics["target_mean"], 1.0)
        self.assertEqual(engine.executor.submissions, 2)
        self.assertEqual(result.training_audit["dataset_fingerprint"], changed.fingerprint())

    def test_score_cache_distinguishes_metric_parameters(self):
        engine = self.make_iaml()
        dataset = Dataset(self.X, self.y)
        self.evaluate(engine, dataset, PrecisionMetric(pos_label=0))
        self.evaluate(engine, dataset, PrecisionMetric(pos_label=1))
        self.assertEqual(engine.executor.submissions, 2)
        self.evaluate(engine, dataset, PrecisionMetric(pos_label=1))
        self.assertEqual(engine.executor.submissions, 2)

    def test_score_cache_uses_main_metric_configuration_selected_by_iaml(self):
        engine = self.make_iaml()
        dataset = Dataset(self.X, np.tile([0, 0, 1], 8))
        for pos_label, submissions in ((0, 1), (1, 2), (1, 2)):
            with self.subTest(pos_label=pos_label, submissions=submissions):
                engine.main_metric = PrecisionMetric(pos_label=pos_label)
                candidate = self.make_candidate(dataset, engine.main_metric)
                candidate.metrics = engine._IAML__metrics_selection(
                    dataset.X, dataset.y, dataset.type_of_target,
                )
                selected = next(metric for metric in candidate.metrics if str(metric) == "precision")
                self.assertIs(selected, engine.main_metric)

                results = engine._IAML__run_evaluations([candidate], dataset, timeout=60)

                self.assertEqual(len(results), 1)
                self.assertEqual(engine.executor.submissions, submissions)
                self.assertIn("precision", results[0].computed_metrics)

    def test_score_cache_distinguishes_groups_and_splitter(self):
        engine = self.make_iaml()
        dataset = Dataset(self.X, self.y)
        self.evaluate(engine, dataset)
        grouped = Dataset(
            self.X, self.y, groups=pd.DataFrame({"group": np.repeat(np.arange(12), 2)}),
        )
        self.evaluate(engine, grouped)
        self.assertEqual(engine.executor.submissions, 2)

        engine.splitter = partial(kfold_splitter, nb_folds=3)
        result = self.evaluate(engine, grouped)
        self.assertEqual(len(result.fold_metrics), 3)
        self.assertEqual(engine.executor.submissions, 3)

    def test_unserializable_splitter_bypasses_score_cache(self):
        engine = self.make_iaml()
        engine.splitter = lambda current: kfold_splitter(current, nb_folds=2)
        dataset = Dataset(self.X, self.y)
        self.evaluate(engine, dataset)
        self.evaluate(engine, dataset)
        self.assertEqual(engine.executor.submissions, 2)
        self.assertFalse(any(key.startswith("IAML_") for key, _ in self.saved))

    def test_stateful_splitter_uses_the_context_before_evaluation(self):
        engine = self.make_iaml()
        engine.splitter = StatefulSplitter()
        dataset = Dataset(self.X, self.y)
        first = self.evaluate(engine, dataset)
        second = self.evaluate(engine, dataset)
        self.assertEqual(len(first.fold_metrics), 2)
        self.assertEqual(len(second.fold_metrics), 3)
        self.assertEqual(engine.executor.submissions, 2)

        compared = engine._IAML__compare_completed_candidates([first, second], dataset, timeout=0)
        self.assertEqual(compared, [second])
        self.assertIs(compared[0], second)
        fallback, population = engine._IAML__previous_population(dataset)
        self.assertEqual(fallback, [second])
        self.assertIs(fallback[0], second)
        self.assertIs(population, dataset)

    def test_draining_results_keeps_the_last_pre_evaluation_splitter_context(self):
        engine = self.make_iaml()
        engine.splitter = StatefulSplitter()
        dataset = Dataset(self.X, self.y)
        result = self.evaluate(engine, dataset)
        context = engine._evaluation_splitter_contexts[dataset.fingerprint()]

        self.assertEqual(engine._IAML__run_evaluations([], dataset, timeout=60), [])
        compared = engine._IAML__compare_completed_candidates([result], dataset, timeout=0)

        self.assertEqual(engine._evaluation_splitter_contexts[dataset.fingerprint()], context)
        self.assertEqual(compared, [result])

    def prepare_stateful_population_comparison(self):
        engine = self.make_iaml()
        engine.splitter = StatefulSplitter()
        reference = Dataset(self.X, self.y.astype(bool))
        subset = reference.sample(12)
        current_candidate = self.make_candidate(reference)
        earlier_candidate = self.make_candidate(subset)
        earlier_candidate.pipeline.predictor[1].configure({'max_depth': 2})
        current = engine._IAML__run_evaluations([current_candidate], reference, timeout=60)[0]
        engine._IAML__run_evaluations([earlier_candidate], subset, timeout=60)
        return engine, reference, current

    def test_final_reevaluation_keeps_only_the_new_successful_context(self):
        engine, reference, current = self.prepare_stateful_population_comparison()

        compared = engine._IAML__compare_completed_candidates([current], reference, timeout=60)

        self.assertEqual(len(current.fold_metrics), 2)
        self.assertEqual(len(compared), 1)
        self.assertEqual(len(compared[0].fold_metrics), 4)
        self.assertTrue(engine._IAML__evaluation_matches(compared[0], reference.fingerprint()))
        self.assertEqual(len(engine._completed_evaluations), 3)

    def test_final_reevaluation_without_score_keeps_the_previous_context(self):
        engine, reference, current = self.prepare_stateful_population_comparison()
        previous_context = engine._evaluation_splitter_contexts[reference.fingerprint()]

        with patch.object(TargetMeanMetric, 'compute', return_value=float('nan')):
            compared = engine._IAML__compare_completed_candidates([current], reference, timeout=60)

        self.assertEqual(compared, [current])
        self.assertIs(compared[0], current)
        self.assertEqual(len(compared[0].fold_metrics), 2)
        self.assertEqual(engine._evaluation_splitter_contexts[reference.fingerprint()], previous_context)
        self.assertTrue(engine._IAML__evaluation_matches(compared[0], reference.fingerprint()))
        self.assertEqual(len(engine._completed_evaluations), 2)
