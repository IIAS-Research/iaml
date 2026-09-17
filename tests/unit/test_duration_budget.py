"""Preparation and submission must consume the evaluation time budget."""

import unittest
from unittest.mock import Mock, patch

import pandas as pd

from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.logger import Logger
from iaml.metrics.accuracy_metric import AccuracyMetric


class TestDurationBudget(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch("iaml.iaml.time.monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", self.previous_verbose)
        self.engine = IAML.__new__(IAML)
        self.engine.executor = Mock()
        self.engine.executor.join.return_value = []
        self.engine.max_stage_duration = 10
        self.engine.splitter = None
        self.engine.keep_training_history = False
        self.dataset = Dataset(pd.DataFrame({"x": range(6)}), [0, 1, 0, 1, 0, 1])
        self.candidates = [
            Candidate(self.dataset, metrics=[AccuracyMetric()], main_metric=AccuracyMetric())
            for _ in range(3)
        ]
        context = patch("iaml.iaml.hash_evaluation_context", return_value="splitter")
        context.start()
        self.addCleanup(context.stop)
        cache_key = patch.object(self.engine, "_IAML__evaluation_cache_key", return_value=None)
        self.cache_key = cache_key.start()
        self.addCleanup(cache_key.stop)

    def evaluate(self, timeout=1, **kwargs):
        return self.engine._IAML__run_evaluations(
            self.candidates, self.dataset, timeout=timeout, **kwargs,
        )

    def test_preparation_and_submission_reduce_join_budget(self):
        def fingerprint():
            self.now += 0.2
            return "data"

        def submit(*args, **kwargs):
            self.now += 0.1
            return True

        self.engine.executor.submit.side_effect = submit
        with patch.object(self.dataset, "fingerprint", side_effect=fingerprint):
            self.evaluate()

        self.assertAlmostEqual(self.engine.executor.join.call_args.args[0], 0.5)
        self.assertEqual(self.engine.executor.submit.call_count, 3)
        for call in self.engine.executor.submit.call_args_list:
            self.assertEqual(call.kwargs["deadline"], 101.0)

    def test_expensive_preparation_leaves_no_time_to_submit(self):
        def fingerprint():
            self.now += 2
            return "data"

        with patch.object(self.dataset, "fingerprint", side_effect=fingerprint):
            self.evaluate()

        self.engine.executor.submit.assert_not_called()
        self.engine.executor.join.assert_called_once_with(0.0)

    def test_stops_submitting_after_budget_is_consumed(self):
        def submit(*args, **kwargs):
            self.now += 1.1
            return True

        self.engine.executor.submit.side_effect = submit
        self.evaluate()

        self.engine.executor.submit.assert_called_once()
        self.engine.executor.join.assert_called_once_with(0.0)

    def test_refused_submission_keeps_completed_results(self):
        completed = self.candidates[0]
        completed.computed_metrics = {"accuracy": 0.75}
        self.engine.executor.join.return_value = [completed]

        def submit(*args, **kwargs):
            self.now = kwargs["deadline"]
            return False

        self.engine.executor.submit.side_effect = submit
        callback = Mock()

        self.assertEqual(self.evaluate(callback=callback), [completed])

        self.engine.executor.submit.assert_called_once()
        self.engine.executor.join.assert_called_once_with(0.0)
        self.assertEqual(callback.call_args.kwargs["remaining_time"], 0.0)
        self.assertEqual(callback.call_args.kwargs["best"], 0.75)

    def test_stage_limit_applies_to_submission_and_join(self):
        for timeout in (None, 30, float("inf")):
            with self.subTest(timeout=timeout):
                self.engine.executor.reset_mock()
                self.evaluate(timeout=timeout)
                self.engine.executor.join.assert_called_once_with(10.0)
                self.assertEqual(self.engine.executor.submit.call_args.kwargs["deadline"], 110.0)

    def test_exhausted_budget_still_joins_previous_work(self):
        for timeout in (0, -0.5):
            with self.subTest(timeout=timeout):
                self.engine.executor.reset_mock()
                self.evaluate(timeout=timeout)
                self.engine.executor.submit.assert_not_called()
                self.engine.executor.join.assert_called_once_with(0.0)

    def test_callback_keeps_global_budget_when_stage_limit_is_shorter(self):
        def fingerprint():
            self.now += 0.2
            return "data"

        callback = Mock()
        with patch.object(self.dataset, "fingerprint", side_effect=fingerprint):
            self.evaluate(timeout=30, callback=callback)

        self.assertAlmostEqual(self.engine.executor.join.call_args.args[0], 9.8)
        self.assertAlmostEqual(callback.call_args.kwargs["remaining_time"], 29.8)

    def test_cache_hits_consume_the_same_budget(self):
        self.cache_key.return_value = "candidate"

        def cached(*args):
            self.now += 0.6
            return {"accuracy": 0.75}

        with patch("iaml.iaml.Cache.from_cache", side_effect=cached):
            results = self.evaluate()

        self.assertEqual(len(results), 2)
        self.assertTrue(all(candidate.get_main_metric_value() == 0.75 for candidate in results))
        self.engine.executor.submit.assert_not_called()
        self.engine.executor.join.assert_called_once_with(0.0)


if __name__ == "__main__":
    unittest.main()
