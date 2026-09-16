"""The Bayesian backend must optimize the same score direction as IAML."""
import unittest

import numpy as np
import pandas as pd

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.actionables.predictors.regressor.act_decision_tree_regressor import (
    ActDecisionTreeRegressor,
)
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metrics import AccuracyMetric, MeanSquaredErrorMetric, R2ScoreMetric
from iaml.optimizers.bayesian_optimizer import BayesianOptimizer


class TestBayesianOptimizerDirection(unittest.TestCase):
    def setUp(self):
        self.previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", self.previous_verbose)

    def make_candidate(self, metric, score, depth):
        X = pd.DataFrame({"feature": np.arange(12, dtype=float) + 0.5})
        if isinstance(metric, AccuracyMetric):
            dataset = Dataset(X, [0, 1] * 6)
            predictor = ActDecisionTreeClassifier()
            estimator_type = "classifier"
        else:
            dataset = Dataset(X, np.linspace(0.1, 2.3, 12))
            predictor = ActDecisionTreeRegressor()
            estimator_type = "regressor"
        predictor.configure("max_depth", depth)
        pipeline = IAMLPipeline(estimator_type=estimator_type)
        pipeline.set_model(predictor)
        candidate = Candidate(dataset, metrics=[metric], main_metric=metric,
                              iaml_pipeline=pipeline)
        candidate.computed_metrics = {str(metric): score}
        return candidate

    def assert_backend_prefers(self, optimizer, depth, score, greater_is_better=True):
        # Inspect the real skopt result, not just the values passed to tell().
        self.assertEqual(len(optimizer.skopt_optimizers), 1)
        backend = next(iter(optimizer.skopt_optimizers.values()))
        result = backend["optimizer"].get_result()
        depth_index = backend["param_keys"].index("0_max_depth")
        self.assertEqual(result.x[depth_index], depth)
        objective = -float(score) if greater_is_better else float(score)
        self.assertAlmostEqual(result.fun, objective)

    def test_backend_selects_the_best_score_and_preserves_candidate_scores(self):
        for metric, worse_score, best_score in (
            (AccuracyMetric(), 0.2, 0.9),
            (R2ScoreMetric(), -4.0, -0.5),
            (R2ScoreMetric(), np.float64(-1.2), np.float32(0.25)),
            (AccuracyMetric(), np.int64(0), np.int64(1)),
            (MeanSquaredErrorMetric(), 77.04, 0.48),
            (MeanSquaredErrorMetric(), np.float64(0.48), np.float32(0.0)),
        ):
            with self.subTest(metric=str(metric), scores=(worse_score, best_score)):
                worse = self.make_candidate(metric, worse_score, depth=1)
                best = self.make_candidate(metric, best_score, depth=3)
                optimizer = BayesianOptimizer()
                optimizer.max_candidates = 2

                outputs = optimizer.run([worse, best])

                self.assert_backend_prefers(
                    optimizer, depth=3, score=best_score,
                    greater_is_better=metric.greater_is_better,
                )
                self.assertIs(outputs[0], best)
                self.assertEqual(len(outputs), 2)
                self.assertEqual(best.computed_metrics, {str(metric): best_score})
                self.assertEqual(worse.computed_metrics, {str(metric): worse_score})
                self.assertGreater(best, worse)

    def test_later_iterations_update_the_best_observation_in_the_same_direction(self):
        metric = AccuracyMetric()
        original = self.make_candidate(metric, 0.9, depth=3)
        worse = self.make_candidate(metric, 0.2, depth=1)
        optimizer = BayesianOptimizer(max_iterations=2)
        optimizer.max_candidates = 2
        optimizer.run([original, worse])
        self.assert_backend_prefers(optimizer, depth=3, score=0.9)
        self.assertFalse(optimizer.finished)

        improved = self.make_candidate(metric, np.float64(0.95), depth=7)
        outputs = optimizer.run([original, improved])

        self.assert_backend_prefers(optimizer, depth=7, score=0.95)
        self.assertIs(outputs[0], improved)
        self.assertEqual(original.get_main_metric_value(), 0.9)
        self.assertEqual(improved.get_main_metric_value(), 0.95)
        self.assertTrue(optimizer.finished)

    def test_later_iterations_recognize_a_lower_loss(self):
        metric = MeanSquaredErrorMetric()
        original = self.make_candidate(metric, 0.48, depth=3)
        worse = self.make_candidate(metric, 77.04, depth=1)
        optimizer = BayesianOptimizer(max_iterations=2)
        optimizer.max_candidates = 2
        optimizer.run([original, worse])
        self.assert_backend_prefers(optimizer, depth=3, score=0.48, greater_is_better=False)

        improved = self.make_candidate(metric, np.float64(0.1), depth=7)
        outputs = optimizer.run([original, improved])

        self.assert_backend_prefers(optimizer, depth=7, score=0.1, greater_is_better=False)
        self.assertIs(outputs[0], improved)
        self.assertEqual(original.get_main_metric_value(), 0.48)
        self.assertEqual(improved.get_main_metric_value(), 0.1)
        self.assertTrue(optimizer.finished)

    def test_candidate_without_main_metric_is_not_a_bayesian_observation(self):
        metric = MeanSquaredErrorMetric()
        evaluated = self.make_candidate(metric, 0.48, depth=3)
        missing = self.make_candidate(metric, 0.0, depth=1)
        missing.computed_metrics = {"r2_score": 0.9}
        optimizer = BayesianOptimizer()
        optimizer.max_candidates = 2

        outputs = optimizer.run([missing, evaluated])

        self.assertIs(outputs[0], evaluated)
        self.assert_backend_prefers(optimizer, depth=3, score=0.48, greater_is_better=False)
        backend = next(iter(optimizer.skopt_optimizers.values()))["optimizer"]
        self.assertEqual(len(backend.get_result().func_vals), 1)
