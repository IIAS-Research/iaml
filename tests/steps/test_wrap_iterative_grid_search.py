"""Exercise iterative search with real components and the current evaluation API."""
from copy import deepcopy
import unittest

import numpy as np

from iaml.actionables.predictors.regressor.act_decision_tree_regressor import (
    ActDecisionTreeRegressor,
)
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.logger import Logger
from iaml.metrics import MeanSquaredErrorMetric
from iaml.wrapper.wrap_iterative_gridsearch import WrapIterativeGridSearch
from tests.helpers.datasets import make_regression_data


class TestWrapIterativeGridSearch(unittest.TestCase):
    def setUp(self):
        previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", previous_verbose)
        self.dataset = Dataset(*make_regression_data(n_samples=36, seed=42))
        self.metric = MeanSquaredErrorMetric()
        self.candidate = Candidate(self.dataset, metrics=[self.metric], main_metric=self.metric)

    def make_step(self, *keys):
        step = ActDecisionTreeRegressor()
        for key, config in step.configuration.items():
            if key not in keys:
                config["no_gridsearch"] = True
        return step

    def test_real_evaluation_explores_depths_and_ranks_the_lowest_loss_first(self):
        step = self.make_step("max_depth", "min_samples_leaf")
        original = deepcopy(step.configuration)
        evaluated = []

        def evaluate(result):
            scores = result.training_evaluate(self.dataset, cache_split=False)
            evaluated.append(result)
            return scores

        wrapper = WrapIterativeGridSearch(step, evaluator=evaluate)
        wrapper.configure({"max_iterations": 3, "patience": 2})
        results = wrapper.run(self.candidate)

        self.assertGreater(len(results), 1)
        self.assertGreater(len({r.pipeline.predictor[1].get_config("max_depth")
                                for r in evaluated}), 1)
        self.assertEqual({r.pipeline.predictor[1].get_config("min_samples_leaf")
                          for r in evaluated}, {1, 2})
        losses = [result.get_main_metric_value() for result in results]
        self.assertTrue(np.isfinite(losses).all())
        self.assertEqual(losses, sorted(losses))
        self.assertEqual(losses[0], min(result.get_main_metric_value() for result in evaluated))
        self.assertEqual(step.configuration, original)
        self.assertFalse(self.candidate.pipeline.have_model)
        self.assertTrue(all(r.pipeline.predictor[1].get_config("random_state") == 42
                            for r in results))
        results[0].pipeline.fit(self.dataset.X, self.dataset.y)
        self.assertTrue(np.isfinite(results[0].predict(self.dataset.X)).all())

    def test_upper_boundary_is_evaluated_and_numeric_range_is_refined(self):
        values = []

        def evaluate(result):
            value = result.pipeline.predictor[1].get_config("max_features")
            values.append(value)
            return {result.main_metric: 1 - value}

        wrapper = WrapIterativeGridSearch(self.make_step("max_features"), evaluator=evaluate)
        wrapper.configure({"max_iterations": 2, "patience": 1})
        results = wrapper.run(self.candidate)

        self.assertTrue(results)
        self.assertIn(1.0, values)
        self.assertIn(0.5, values)
        self.assertTrue(any(0.5 < value < 1.0 for value in values))
        self.assertTrue(all(0.1 <= value <= 1.0 for value in values))
        self.assertLess(len(values), 30)

    def test_categories_are_evaluated_once_even_with_low_patience(self):
        step = self.make_step("criterion")
        visited = []

        def evaluate(result):
            visited.append(result.pipeline.predictor[1].get_config("criterion"))
            return {result.main_metric: 1.0}

        wrapper = WrapIterativeGridSearch(step, evaluator=evaluate)
        wrapper.configure("patience", 1)
        results = wrapper.run(self.candidate)

        self.assertEqual(visited, step.configuration["criterion"]["categorical"])
        self.assertEqual(len(results), len(visited))

    def test_iteration_limit_reaches_nested_parameters(self):
        visited = []

        def evaluate(result):
            visited.append(result.pipeline.predictor[1].resume_configuration())
            return {result.main_metric: 1.0}

        step = self.make_step("max_depth", "min_samples_leaf")
        wrapper = WrapIterativeGridSearch(step, evaluator=evaluate)
        wrapper.configure("max_iterations", 1)
        results = wrapper.run(self.candidate)

        self.assertEqual(len(results), 1)
        self.assertEqual(visited, [step.resume_configuration()])

    def test_no_searchable_parameters_runs_once_without_an_evaluator(self):
        step = self.make_step("random_state")
        wrapper = WrapIterativeGridSearch(step)
        results = wrapper.run(self.candidate)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].pipeline.predictor[1].resume_configuration(),
                         step.resume_configuration())
        self.assertFalse(self.candidate.pipeline.have_model)

    def test_search_requires_scores_instead_of_calling_obsolete_evaluate(self):
        wrapper = WrapIterativeGridSearch(self.make_step("max_depth"))
        with self.assertRaisesRegex(ValueError, "evaluator"):
            wrapper.run(self.candidate)

    def test_failed_evaluations_do_not_become_successful_candidates(self):
        for score in (None, np.nan):
            with self.subTest(score=score):
                wrapper = WrapIterativeGridSearch(
                    self.make_step("max_depth"),
                    evaluator=lambda result: {result.main_metric: score},
                )
                wrapper.configure("max_iterations", 1)
                self.assertEqual(wrapper.run(self.candidate), [])

    def test_repeated_run_uses_current_evaluator_context(self):
        score = 1.0
        wrapper = WrapIterativeGridSearch(
            self.make_step("max_depth"), evaluator=lambda result: {result.main_metric: score},
        )
        wrapper.configure("max_iterations", 1)
        self.assertEqual(wrapper.run(self.candidate)[0].get_main_metric_value(), 1.0)
        score = 2.0
        self.assertEqual(wrapper.run(self.candidate)[0].get_main_metric_value(), 2.0)

    def test_pipeline_round_trip_accepts_the_evaluator(self):
        evaluator = lambda result: result.training_evaluate(self.dataset, cache_split=False)
        step = ActDecisionTreeRegressor()
        step.configure("max_depth", 3)
        wrapper = WrapIterativeGridSearch(step, evaluator=evaluator)
        restored = WrapIterativeGridSearch.from_pipeline(wrapper.json_pipeline(), evaluator=evaluator)
        self.assertIs(restored.evaluator, evaluator)
        self.assertEqual(restored.step.resume_configuration(), wrapper.step.resume_configuration())
