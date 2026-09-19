"""Warmup uses the production stage deadline and never bypasses bounded CV."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import ActDecisionTreeClassifier
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metrics.accuracy_metric import AccuracyMetric


class WarmupBudgetTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.generation_seconds = 0.0
        self.clock = patch("iaml.iaml.time.monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        logger = Logger()
        self.addCleanup(setattr, logger, "verbose", logger.verbose)
        logger.verbose = 0
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        cache.configure(None)
        self.X = pd.DataFrame({"number": np.arange(24, dtype=float)})
        self.y = np.tile([False, True], 12)
        self.metric = AccuracyMetric()
        self.minimal = self.make_candidate("minimal", 1)
        self.normal = self.make_candidate("normal", 2)
        self.minimal_pool = [self.minimal]
        self.normal_pool = [self.normal]
        # Exercise real fit orchestration and __run_evaluations. Candidate
        # generation, worker completion and final estimator fit are simulated;
        # no model is trained and no process or wall-clock sleep is needed.
        self.engine = IAML.__new__(IAML)
        for name, value in {
            "max_duration": 60, "max_stage_duration": 15, "max_workers": 1,
            "train_on_n_samples": None, "refit_on_sample": True,
            "initial_preprocessor": None, "main_metric": self.metric,
            "splitter": None, "keep_training_history": True,
            "time_before_sample_use": 60, "executor": None,
        }.items():
            setattr(self.engine, name, value)
        self.executor = Mock()
        self.executor.submit.side_effect = self.submit
        self.executor.join.side_effect = self.join
        self.queued = []
        self.stages = []
        self.responses = []
        self.callback = Mock()

    def make_candidate(self, name, depth):
        candidate = Candidate(Dataset(self.X, self.y), metrics=[self.metric], main_metric=self.metric)
        predictor = ActDecisionTreeClassifier()
        predictor.configure({"max_depth": depth, "random_state": 42})
        candidate.pipeline.steps = [(name, predictor)]
        return candidate

    def submit(self, target, candidate, dataset, **options):
        self.queued.append((candidate, options))
        return True

    def join(self, timeout):
        self.stages.append({"started": self.now, "timeout": timeout, "queued": list(self.queued)})
        requested_duration, scores = self.responses.pop(0)
        self.now += min(requested_duration, timeout)
        completed = []
        for candidate, options in self.queued:
            name = candidate.pipeline.predictor[0]
            if name not in scores:
                continue
            # Independent returned instances match the subprocess contract.
            result = deepcopy(candidate)
            score = scores[name]
            result.computed_metrics = {} if score is None else {result.main_metric: score}
            result.training_audit = {
                "pipeline_fingerprint": result.pipeline.fingerprint(),
                "dataset_fingerprint": "training-population", "status": "success",
                "error": None, "metrics": deepcopy(result.computed_metrics),
            }
            completed.append(result)
        self.queued.clear()
        return completed

    def fit(self):
        def generate(candidate):
            self.now += self.generation_seconds
            return self.normal_pool

        with (
            patch.object(self.engine, "check_pipeline"),
            patch.object(self.engine, "_IAML__metrics_selection", return_value=[self.metric]),
            patch.object(self.engine, "_IAML__generate_minimal_candidates", return_value=self.minimal_pool),
            patch.object(self.engine, "_IAML__run", side_effect=generate),
            patch.object(self.engine, "_IAML__build_optimizer", return_value=Mock()),
            patch.object(self.engine, "_IAML__optimize", side_effect=lambda data, candidates, **kwargs: candidates) as optimize,
            patch("iaml.iaml.TimedPoolExecutor", return_value=self.executor),
            patch("iaml.iaml.hash_evaluation_context", return_value=None),
            patch.object(Cache, "reset"),
            patch.object(Candidate, "training_evaluate", side_effect=AssertionError("Unbounded parent evaluation")) as direct,
            patch.object(IAMLPipeline, "fit", autospec=True) as final_fit,
        ):
            self.optimize, self.direct, self.final_fit = optimize, direct, final_fit
            return self.engine.fit(self.X, self.y, verbose=0, callback=self.callback, n_candidates=1)

    def test_minimalist_warmup_uses_stage_deadline_and_keeps_full_search_pool(self):
        self.generation_seconds = 12
        self.responses = [(0.5, {"minimal": 0.7}), (1, {"normal": 0.8})]
        result = self.fit()
        self.assertEqual([item.pipeline.predictor[0] for item, _ in self.stages[0]["queued"]], ["minimal"])
        self.assertEqual([item.pipeline.predictor[0] for item, _ in self.stages[1]["queued"]], ["minimal", "normal"])
        self.assertAlmostEqual(self.stages[0]["timeout"], 9.6)
        self.assertAlmostEqual(self.stages[0]["queued"][0][1]["deadline"], 21.6)
        self.assertEqual(self.callback.call_args_list[0].kwargs["remaining_time"], 47.5)
        self.assertEqual(self.optimize.call_args.kwargs["max_duration"], 46.5)
        self.assertEqual(result[0].pipeline.predictor[0], "normal")
        self.assertEqual(len(self.engine.training_history), 2)
        self.direct.assert_not_called()
        self.final_fit.assert_called_once()

    def test_remaining_global_budget_can_be_shorter_than_stage_limit(self):
        self.generation_seconds = 55
        self.responses = [(100, {}), (100, {})]
        with self.assertRaises(TimeoutError):
            self.fit()
        self.assertEqual(self.stages[0]["timeout"], 1)
        self.assertEqual(self.stages[0]["queued"][0][1]["deadline"], 56)
        self.assertEqual(self.stages[1]["timeout"], 4)
        self.assertEqual(self.now, 60)
        self.final_fit.assert_not_called()
        self.direct.assert_not_called()

    def test_exhausted_generation_budget_does_not_start_unbounded_warmup(self):
        self.generation_seconds = 61
        with self.assertRaises(TimeoutError):
            self.fit()
        self.executor.submit.assert_not_called()
        self.executor.join.assert_not_called()
        self.final_fit.assert_not_called()
        self.direct.assert_not_called()

    def test_timed_out_warmup_leaves_all_candidates_for_initial_evaluation(self):
        self.responses = [(100, {}), (1, {"normal": 0.75})]
        result = self.fit()
        self.assertEqual(self.stages[0]["timeout"], 12)
        self.assertEqual(self.stages[1]["started"], 12)
        self.assertEqual([item.pipeline.predictor[0] for item, _ in self.stages[1]["queued"]], ["normal", "minimal"])
        self.assertEqual(result[0].pipeline.predictor[0], "normal")
        self.assertEqual(self.callback.call_args_list[0].kwargs["generation_size"], 0)
        self.assertEqual(self.callback.call_args_list[1].kwargs["generation_size"], 1)

    def test_completed_warmup_can_be_refitted_after_initial_evaluation_times_out(self):
        self.generation_seconds = 55
        self.responses = [(0.5, {"minimal": 0.0}), (100, {})]
        result = self.fit()
        self.assertEqual(result[0].pipeline.predictor[0], "minimal")
        self.assertEqual(result[0].get_main_metric_value(), 0.0)
        self.assertEqual(len(self.stages), 2)
        self.assertEqual(self.now, 60)
        self.final_fit.assert_called_once()

    def test_incomplete_warmup_metrics_never_become_fallback_scores(self):
        self.generation_seconds = 55
        self.responses = [(0.5, {"minimal": None}), (100, {})]
        with self.assertRaises(TimeoutError):
            self.fit()
        self.assertEqual(self.callback.call_args.kwargs["generation_size"], 0)
        self.final_fit.assert_not_called()

    def test_without_minimalists_normal_warmup_still_uses_bounded_executor(self):
        self.minimal_pool = []
        self.responses = [(0.5, {"normal": 0.7}), (1, {"normal": 0.7})]
        self.fit()
        self.assertEqual(self.stages[0]["queued"][0][0].pipeline.predictor[0], "normal")
        self.assertEqual(self.stages[0]["timeout"], 12)
        self.direct.assert_not_called()

    def test_default_stage_limit_cannot_give_all_remaining_time_to_warmup(self):
        self.engine.max_stage_duration = IAML(max_duration=60, max_workers=1).max_stage_duration
        self.responses = [(100, {}), (1, {"normal": 0.75})]
        result = self.fit()
        self.assertEqual(self.stages[0]["timeout"], 12)
        self.assertEqual(self.stages[1]["timeout"], 48)
        self.assertEqual(result[0].pipeline.predictor[0], "normal")

    def test_shorter_explicit_stage_limit_still_caps_warmup(self):
        self.engine.max_stage_duration = 3
        self.responses = [(100, {}), (1, {"normal": 0.75})]
        self.fit()
        self.assertEqual(self.stages[0]["timeout"], 3)
        self.assertEqual(self.stages[1]["started"], 3)

    def test_unlimited_search_still_bounds_warmup_by_stage_duration(self):
        self.engine.max_duration = -1
        self.responses = [(0.5, {"minimal": 0.7}), (1, {"normal": 0.7})]
        self.fit()
        self.assertEqual(self.stages[0]["timeout"], 15)
        self.assertEqual(self.stages[0]["queued"][0][1]["deadline"], 15)
        self.assertEqual(self.optimize.call_args.kwargs["max_duration"], float("inf"))


if __name__ == "__main__":
    unittest.main()
