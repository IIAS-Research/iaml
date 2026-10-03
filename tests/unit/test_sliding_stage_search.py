"""Sliding search stages retain pending evaluations and their original context."""
from copy import deepcopy
from functools import partial
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import ActDecisionTreeClassifier
from iaml.cache import Cache
from iaml.cache_keys import hash_evaluation_context
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.logger import Logger
from iaml.metrics.accuracy_metric import AccuracyMetric
from iaml.shared_cache import CacheService
from iaml.splitters import kfold_splitter


class StageExecutor:
    """Control worker completion without relying on process timing."""

    sliding_stages = True

    def __init__(self, test):
        self.test = test
        self.pending = []
        self.submissions = []
        self.waits = []
        self.completions = []
        self.accept_limit = None
        self.callback = None

    def set_callback(self, callback):
        self.callback = callback

    def submit(self, target, candidate, dataset, **options):
        if self.accept_limit is not None and len(self.submissions) >= self.accept_limit:
            self.test.now = options['deadline']
            return False
        job = candidate, dataset, options, self.callback
        self.pending.append(job)
        self.submissions.append(job)
        return True

    def join(self, timeout, *, cancel_pending=None):
        self.waits.append((timeout, cancel_pending))
        duration, scores = self.completions.pop(0)
        self.test.now += min(duration, timeout)
        results, pending = [], []
        for candidate, dataset, options, callback in self.pending:
            name = candidate.pipeline.predictor[0]
            if name not in scores:
                pending.append((candidate, dataset, options, callback))
                continue
            result = deepcopy(candidate)
            result.computed_metrics = {result.main_metric: scores[name]}
            result.training_audit = result.build_training_audit(
                dataset=dataset, fold_metrics=[], aggregated_metrics=result.computed_metrics,
                status='success')
            returned = options['evaluation_id'], result
            callback(returned)
            results.append(returned)
        self.pending = [] if cancel_pending and duration >= timeout else pending
        return results


class SlidingStageSearchTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        clock = patch('iaml.iaml.time.monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        logger = Logger()
        self.addCleanup(setattr, logger, 'verbose', logger.verbose)
        logger.verbose = 0
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        self.saved = {}
        backend = CacheService()
        backend.__set_backend__(self.saved, [], SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)
        self.dataset = Dataset(pd.DataFrame({'x': range(12)}), [0, 1] * 6)
        self.engine = IAML.__new__(IAML)
        self.engine.executor = StageExecutor(self)
        self.engine.max_stage_duration = 1
        self.engine.splitter = partial(kfold_splitter, nb_folds=2)
        self.engine.keep_training_history = True
        self.engine.training_history = []
        self.engine._training_history_seen = set()

    def candidate(self, name, depth):
        metric = AccuracyMetric()
        candidate = Candidate(self.dataset, metrics=[metric], main_metric=metric)
        model = ActDecisionTreeClassifier()
        model.configure({'max_depth': depth})
        candidate.pipeline.steps = [(name, model)]
        return candidate

    def evaluate(self, candidates, dataset=None, **options):
        return self.engine._IAML__run_evaluations(
            candidates, self.dataset if dataset is None else dataset,
            timeout=10 - self.now, **options)

    def test_repeated_pending_candidates_run_once_and_return_in_a_later_stage(self):
        fast, slow = self.candidate('fast', 1), self.candidate('slow', 2)
        executor = self.engine.executor
        executor.completions = [(1, {'fast': 0.6}), (1, {'slow': 0.9}), (0, {})]
        first = self.evaluate([fast, slow])
        second = self.evaluate([deepcopy(slow)])
        third = self.evaluate([])
        self.assertEqual([item.pipeline.predictor[0] for item in first], ['fast'])
        self.assertEqual([item.pipeline.predictor[0] for item in second], ['slow'])
        self.assertEqual(third, [])
        self.assertEqual(len(executor.submissions), 2)
        self.assertEqual(executor.waits[:2], [(1, False), (1, False)])
        self.assertEqual(len(self.engine.training_history), 2)
        self.assertFalse(self.engine._IAML__evaluations_pending())

    def test_unpickleable_splitter_still_deduplicates_pending_candidates(self):
        self.engine.splitter = lambda dataset: kfold_splitter(dataset, nb_folds=2)
        slow = self.candidate('slow', 2)
        executor = self.engine.executor
        executor.completions = [(1, {}), (1, {'slow': 0.9})]
        self.assertEqual(self.evaluate([slow]), [])
        self.assertEqual(len(self.evaluate([deepcopy(slow)])), 1)
        self.assertEqual(len(executor.submissions), 1)
        self.assertFalse(any(key.startswith('IAML_') for key, _ in self.saved))

    def test_candidates_not_submitted_before_deadline_are_preserved(self):
        candidates = [self.candidate(name, depth) for depth, name in enumerate('abc', 1)]
        executor = self.engine.executor
        executor.accept_limit = 1
        executor.completions = [(0, {}), (1, dict(a=0.6, b=0.7, c=0.8))]
        self.assertEqual(self.evaluate(candidates), [])
        self.assertEqual(len(self.engine._evaluation_queue), 2)
        executor.accept_limit = None
        completed = self.evaluate([])
        self.assertEqual([candidate.pipeline.predictor[0] for candidate in completed], ['c', 'b', 'a'])
        self.assertEqual(len(executor.submissions), 3)
        self.assertFalse(self.engine._IAML__evaluations_pending())

    def test_late_result_keeps_its_original_dataset_cache_key(self):
        slow, newer = self.candidate('slow', 2), self.candidate('newer', 3)
        changed = Dataset(self.dataset.X, [1] * 12)
        executor = self.engine.executor
        executor.completions = [(1, {}), (1, {'slow': 0.9, 'newer': 0.8}), (0, {})]
        self.assertEqual(self.evaluate([slow]), [])
        self.assertEqual(len(self.evaluate([newer], dataset=changed)), 2)
        key = self.engine._IAML__evaluation_cache_key(
            slow, hash_evaluation_context(self.engine.splitter))
        self.assertIsNotNone(Cache().from_cache(key, self.dataset.fingerprint()))
        self.assertIsNone(Cache().from_cache(key, changed.fingerprint()))
        cached = self.evaluate([deepcopy(slow)])
        self.assertEqual(cached[0].get_main_metric_value(), 0.9)
        self.assertEqual(len(executor.submissions), 2)

    def test_global_deadline_cancels_pending_work(self):
        executor = self.engine.executor
        executor.completions = [(1, {}), (100, {})]
        self.evaluate([self.candidate('slow', 2)])
        self.now = 9.5
        self.assertEqual(self.evaluate([]), [])
        self.assertEqual(executor.waits[-1], (0.5, True))
        self.assertEqual(executor.pending, [])
        self.assertFalse(self.engine._IAML__evaluations_pending())

    def test_terminal_drain_uses_global_budget_instead_of_stage_limit(self):
        executor = self.engine.executor
        executor.completions = [(1, {}), (4, {'slow': 0.9})]
        self.evaluate([self.candidate('slow', 2)])
        completed = self.evaluate([], drain_pending=True)
        self.assertEqual(executor.waits[-1], (9, True))
        self.assertEqual(completed[0].get_main_metric_value(), 0.9)
        self.assertEqual(self.now, 5)

    def test_empty_stage_does_not_spend_patience_or_discard_pending_result(self):
        fast, slow = self.candidate('fast', 1), self.candidate('slow', 2)
        fast.computed_metrics = {'accuracy': 0.6}
        optimizer = Mock()
        optimizer.finished = False
        optimizer.run.return_value = [slow]
        executor = self.engine.executor
        executor.completions = [(1, {}), (1, {'slow': 0.9}), (0, {})]
        completed = self.engine._IAML__optimize(
            self.dataset, [fast], optimizer=optimizer, max_duration=10, patience=1)
        self.assertEqual([candidate.pipeline.predictor[0] for candidate in completed], ['slow', 'fast'])
        self.assertEqual(optimizer.run.call_count, 3)
        self.assertEqual(len(executor.submissions), 1)


if __name__ == '__main__':
    unittest.main()
