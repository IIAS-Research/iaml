"""The Bayesian backend must preserve IAML's score direction and parameter domains."""
import unittest

import numpy as np
import pandas as pd
from skopt.space import Categorical

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.actionables.predictors.classifier.act_hist_gradient_boosting_classifier import (
    ActHistGradientBoostingClassifier,
)
from iaml.actionables.predictors.classifier.act_randomforest import ActRandomForest
from iaml.actionables.predictors.regressor.act_decision_tree_regressor import (
    ActDecisionTreeRegressor,
)
from iaml.actionables.predictors.survival.act_random_survival_forest import ActRandomSurvivalForest
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metrics import AccuracyMetric, MeanSquaredErrorMetric, R2ScoreMetric
from iaml.optimizers.bayesian_optimizer import BayesianOptimizer
from tests.helpers.datasets import make_classification_data, make_survival_data


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


class TestBayesianOptimizerParameters(unittest.TestCase):
    def setUp(self):
        previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", previous_verbose)

    def make_candidate(self, predictor, survival=False):
        make_data = make_survival_data if survival else make_classification_data
        candidate = Candidate(Dataset(*make_data(n_samples=24, seed=42)))
        candidate.pipeline.set_model(predictor)
        candidate.computed_metrics = {candidate.main_metric: 0.5}
        return candidate

    def optimize(self, candidate):
        optimizer = BayesianOptimizer()
        optimizer.max_candidates = 2
        outputs = optimizer.run([candidate])
        self.assertEqual(len(outputs), 2)
        self.assertIs(outputs[0], candidate)
        backend = next(iter(optimizer.skopt_optimizers.values()))
        self.assertEqual(len(backend["param_keys"]), len(backend["dimensions"]))
        self.assertEqual(len(backend["optimizer"].Xi), 1)
        return outputs[1], backend

    def test_bootstrap_is_boolean_in_observations_and_suggestions(self):
        for value in (False, True, np.bool_(False), np.bool_(True)):
            with self.subTest(value=value, value_type=type(value)):
                model = ActRandomForest()
                model.configure("bootstrap", value)
                candidate = self.make_candidate(model)
                suggested, backend = self.optimize(candidate)
                index = backend["param_keys"].index("0_bootstrap")
                dimension = backend["dimensions"][index]
                self.assertIsInstance(dimension, Categorical)
                self.assertEqual(set(dimension.categories), {True, False})
                self.assertIs(backend["optimizer"].Xi[0][index], bool(value))
                self.assertIs(type(suggested.pipeline.predictor[1].get_config("bootstrap")), bool)
                self.assertIs(model.get_config("bootstrap"), value)

    def test_survival_forest_keeps_unbounded_depth_and_optimizes_other_parameters(self):
        for depth in (None, 4):
            with self.subTest(depth=depth):
                model = ActRandomSurvivalForest()
                model.configure("max_depth", depth)
                candidate = self.make_candidate(model, survival=True)
                suggested, backend = self.optimize(candidate)
                self.assertEqual(backend["param_keys"], [
                    "0_min_samples_leaf", "0_min_samples_split", "0_n_estimators",
                ])
                self.assertEqual(backend["optimizer"].Xi[0], [1, 2, 100])
                proposed_model = suggested.pipeline.predictor[1]
                self.assertEqual(proposed_model.get_config("max_depth"), depth)
                for key, dimension in zip(backend["param_keys"], backend["dimensions"]):
                    self.assertIn(proposed_model.get_config(key[2:]), dimension)
                # Exercise the generated configuration with the real survival estimator.
                proposed_model.fit(candidate.dataset)
                self.assertEqual(proposed_model.model.max_depth, depth)

    def test_numeric_categories_and_none_are_not_treated_as_numeric_bounds(self):
        for depth in (None, 5, 15):
            with self.subTest(depth=depth):
                model = ActHistGradientBoostingClassifier()
                model.configure("max_depth", depth)
                suggested, backend = self.optimize(self.make_candidate(model))
                index = backend["param_keys"].index("0_max_depth")
                dimension = backend["dimensions"][index]
                self.assertIsInstance(dimension, Categorical)
                self.assertEqual(dimension.categories, (None, 3, 5, 10, 15))
                self.assertEqual(backend["optimizer"].Xi[0][index], depth)
                self.assertIn(suggested.pipeline.predictor[1].get_config("max_depth"), dimension)

    def test_candidate_without_search_dimensions_is_preserved(self):
        candidate = self.make_candidate(ActRandomSurvivalForest(), survival=True)
        optimizer = BayesianOptimizer()
        optimizer.ignored_configs.update({"min_samples_leaf", "min_samples_split", "n_estimators"})
        outputs = optimizer.run([candidate])
        self.assertEqual(len(outputs), 1)
        self.assertIs(outputs[0], candidate)
        self.assertEqual(optimizer.skopt_optimizers, {})
