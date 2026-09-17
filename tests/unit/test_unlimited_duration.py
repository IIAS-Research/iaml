"""Exercise the default duration through real generation, CV and optimization."""

from functools import partial
import math
import random
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.iaml import IAML
from iaml.logger import Logger
from iaml.meta_explorer_step import MetaExplorerStep
from iaml.metrics.accuracy_metric import AccuracyMetric
from iaml.optimizers.genetic_optimizer import GeneticOptimizer
from iaml.splitters import kfold_splitter
from iaml.timed_pool_executor import TimedPoolExecutor
from iaml.worker_manager import WorkerManager


class TestUnlimitedDuration(unittest.TestCase):
    def setUp(self):
        previous_verbose = Logger().verbose
        self.addCleanup(setattr, Logger(), "verbose", previous_verbose)
        previous_random = random.getstate()
        random.seed(20)
        self.addCleanup(random.setstate, previous_random)
        workers = WorkerManager(max_workers=1)
        previous_workers = workers.max_workers
        workers.max_workers = 1
        self.addCleanup(setattr, workers, "max_workers", previous_workers)

    def run_search(self, **duration):
        optimizers = []

        def optimizer_factory(duration=None):
            optimizer = GeneticOptimizer(nb_candidate=4, duration=duration)
            optimizer.max_generations = 1
            optimizers.append(optimizer)
            return optimizer

        engine = IAML(
            **duration, max_workers=1, max_stage_duration=5,
            splitter=partial(kfold_splitter, nb_folds=2),
            main_metric=AccuracyMetric(), optimizer=optimizer_factory,
        )
        # Limit the real search to two small trees, with no interchangeable models.
        engine.first_step = MetaExplorerStep()
        for depth in (1, 2):
            model = ActDecisionTreeClassifier()
            model.configure({"max_depth": depth})
            engine.first_step.add_step(model)
            model.is_interchangeable = False
        engine.minimal_predictor_step = None

        def executor_factory(*args, **kwargs):
            executor = TimedPoolExecutor(*args, **kwargs)
            self.addCleanup(executor.shutdown)
            if not executor._mp_capable or executor.debug:
                self.skipTest("Multiprocessing IPC is unavailable")
            return executor

        X = pd.DataFrame({"feature": np.linspace(-1, 1, 24)})
        y = (X["feature"] >= 0).astype(int).to_numpy()
        stages = []
        with patch("iaml.iaml.TimedPoolExecutor", side_effect=executor_factory):
            results = engine.fit(X, y, callback=lambda **stage: stages.append(stage), verbose=-1)

        self.assertEqual(len(results), 1)
        self.assertTrue(math.isfinite(results[0].get_main_metric_score()))
        self.assertEqual(results[0].predict(X).shape, y.shape)
        self.assertEqual(len(engine.candidates), 2)
        return stages, optimizers[0]

    def assert_search_ran(self, stages, optimizer):
        self.assertEqual([stage["generation"] for stage in stages], [None, 0])
        self.assertEqual(stages[0]["generation_size"], 2)
        self.assertGreater(stages[1]["generation_size"], 0)
        self.assertEqual(optimizer.generation_count, 1)

    def test_default_duration_evaluates_and_optimizes(self):
        stages, optimizer = self.run_search()
        self.assert_search_ran(stages, optimizer)
        self.assertIsNone(optimizer.duration)

    def test_explicit_unlimited_duration_evaluates_and_optimizes(self):
        stages, optimizer = self.run_search(max_duration=-1)
        self.assert_search_ran(stages, optimizer)
        self.assertIsNone(optimizer.duration)

    def test_finite_duration_still_evaluates_and_optimizes(self):
        stages, optimizer = self.run_search(max_duration=10)
        self.assert_search_ran(stages, optimizer)
        self.assertGreater(optimizer.duration, 0)
        self.assertLessEqual(optimizer.duration, 10)

    def test_expired_finite_budget_does_not_start_optimization(self):
        stages, optimizer = self.run_search(max_duration=0)
        self.assertEqual(stages, [])
        self.assertEqual(optimizer.generation_count, 0)
        self.assertEqual(optimizer.duration, 0)


if __name__ == "__main__":
    unittest.main()
