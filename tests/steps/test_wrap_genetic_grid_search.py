"""Tests for WrapGeneticGridSearch."""
from copy import deepcopy
import random
from unittest.mock import patch

import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.regressor.act_decision_tree_regressor import (
    ActDecisionTreeRegressor,
)
from iaml.candidate import Candidate
from iaml.decorators.all import is_step, runner
from iaml.metrics.r2_score_metric import R2ScoreMetric
from iaml.step import Step
from iaml.wrapper.wrap_genetic_gridsearch import WrapGeneticGridSearch


@is_step("test")
class DummyGeneticStep(Step):
    name = "DummyGeneticStep"

    def __init__(self, configuration: dict) -> None:
        self.configuration = configuration
        self.run_called = False

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        self.run_called = True
        new_candidate = candidate.to_output()
        new_candidate.config_snapshot = self.resume_configuration()
        return new_candidate


@is_step("test")
class ScoredGeneticStep(Step):
    """Assign a deterministic score without constructing an obsolete stack."""

    def __init__(self, configuration: dict) -> None:
        self.configuration = configuration

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        result = candidate.to_output()
        result.config_snapshot = self.resume_configuration()
        result.computed_metrics = {result.main_metric: self.get_config("score")}
        return result


@is_step("test")
class EmptyGeneticStep(Step):
    """Represent a worker that cannot produce any candidate."""

    def __init__(self) -> None:
        self.configuration = {"alpha": {"default": 1, "range": [1, 2]}}

    @runner
    def run(self, candidate: Candidate) -> list[Candidate]:
        return []


class TestWrapGeneticGridSearch(StepTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.addCleanup(random.setstate, random.getstate())
        random.seed(42)

    def _make_candidate(self) -> Candidate:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dataset = self.make_dataset(df, y=[0, 1])
        return Candidate(dataset)

    def test_run_short_circuits_when_no_configurable_keys(self) -> None:
        config = {
            "random_state": {"value": 7, "default": 7},
            "locked": {"value": 1.0, "default": 1.0, "no_gridsearch": True},
        }
        step = DummyGeneticStep(config)
        wrapper = WrapGeneticGridSearch(step)

        results = wrapper.run(self._make_candidate())

        self.assertTrue(step.run_called)
        self.assertEqual(len(results), 1)
        snapshot = results[0].config_snapshot
        self.assertEqual(snapshot["random_state"], 7)
        self.assertEqual(snapshot["locked"], 1.0)

    def test_random_generation_respects_ranges_and_ignores_random_state(self) -> None:
        config = {
            "alpha": {"value": 10, "default": 10, "range": [1, 20]},
            "mode": {"value": "fast", "default": "fast", "categorical": ["fast", "slow"]},
            "use_bias": {"value": True, "default": True},
            "random_state": {"value": 123, "default": 123},
            "locked": {"value": 1.0, "default": 1.0, "no_gridsearch": True},
        }
        step = DummyGeneticStep(config)
        wrapper = WrapGeneticGridSearch(step)

        generated = wrapper.random_generation()
        snapshot = generated.resume_configuration()

        self.assertEqual(snapshot["random_state"], 123)
        self.assertEqual(snapshot["locked"], 1.0)
        self.assertIn(snapshot["mode"], {"fast", "slow"})
        self.assertIsInstance(snapshot["use_bias"], bool)
        self.assertTrue(1 <= snapshot["alpha"] <= 20)

    def test_random_mutation_changes_numeric_value_when_no_delta(self) -> None:
        config = {
            "alpha": {"value": 3, "default": 3, "range": [1, 10]},
            "random_state": {"value": 11, "default": 11},
            "locked": {"value": 2.0, "default": 2.0, "no_gridsearch": True},
        }
        step = DummyGeneticStep(config)
        wrapper = WrapGeneticGridSearch(step)

        with patch("iaml.wrapper.wrap_genetic_gridsearch.random.choice") as mock_choice, \
            patch("iaml.wrapper.wrap_genetic_gridsearch.random.uniform", return_value=0):
            mock_choice.side_effect = ["alpha", 1]
            mutated = wrapper.random_mutation(step)

        first_choice_options = mock_choice.call_args_list[0].args[0]
        self.assertNotIn("random_state", first_choice_options)
        self.assertNotIn("locked", first_choice_options)
        self.assertEqual(step.get_config("alpha"), 3)
        self.assertEqual(mutated.get_config("alpha"), 4)
        self.assertEqual(mutated.get_config("random_state"), 11)
        self.assertEqual(mutated.get_config("locked"), 2.0)

    def test_random_mutation_respects_categorical_and_range_limits(self) -> None:
        config = {
            "alpha": {"value": 10, "default": 10, "range": [1, 10]},
            "mode": {"value": "fast", "default": "fast", "categorical": ["fast", "slow"]},
        }
        step = DummyGeneticStep(config)
        wrapper = WrapGeneticGridSearch(step)

        with patch("iaml.wrapper.wrap_genetic_gridsearch.random.choice") as mock_choice, \
            patch("iaml.wrapper.wrap_genetic_gridsearch.random.uniform", return_value=0.1):
            mock_choice.side_effect = ["alpha", "mode", "slow"]
            mutated_numeric = wrapper.random_mutation(step)
            mutated_categorical = wrapper.random_mutation(step)

        self.assertEqual(mutated_numeric.get_config("alpha"), 10)
        self.assertIn(mutated_categorical.get_config("mode"), {"fast", "slow"})

    def test_run_single_generation_returns_expected_candidates(self) -> None:
        config = {
            "alpha": {"value": 10, "default": 10, "range": [1, 20]},
            "mode": {"value": "fast", "default": "fast", "categorical": ["fast", "slow"]},
            "use_bias": {"value": True, "default": True},
            "random_state": {"value": 123, "default": 123},
        }
        step = DummyGeneticStep(config)
        wrapper = WrapGeneticGridSearch(step)
        wrapper.configure({"nb_generations": 1, "nb_estimators": 3})

        results = wrapper.run(self._make_candidate())

        self.assertEqual(len(results), 3)
        snapshots = [candidate.config_snapshot for candidate in results]
        self.assertEqual({snap["random_state"] for snap in snapshots}, {123})
        for snap in snapshots:
            self.assertIn(snap["mode"], {"fast", "slow"})
            self.assertIn(snap["use_bias"], {True, False})
            self.assertTrue(1 <= snap["alpha"] <= 20)

    def test_numeric_categories_take_precedence_over_multipliers(self) -> None:
        step = DummyGeneticStep({"alpha": {"default": 4, "categorical": [4, 9]}})
        wrapper = WrapGeneticGridSearch(step)

        with patch("iaml.wrapper.wrap_genetic_gridsearch.random.choice") as choice, \
            patch("iaml.wrapper.wrap_genetic_gridsearch.random.getrandbits", return_value=0), \
            patch("iaml.wrapper.wrap_genetic_gridsearch.random.uniform", return_value=2):
            choice.side_effect = [9, "alpha", 9]
            generated = wrapper.random_generation()
            mutated = wrapper.random_mutation(step)

        self.assertEqual(generated.get_config("alpha"), 9)
        self.assertEqual(mutated.get_config("alpha"), 9)
        self.assertEqual(step.get_config("alpha"), 4)

    def _scored_step(self, score: int) -> ScoredGeneticStep:
        return ScoredGeneticStep({
            "fixed": {"default": 1, "no_gridsearch": True},
            "score": {"default": score, "range": [0, 10]},
        })

    def test_selection_keeps_best_candidate_without_legacy_stack(self) -> None:
        wrapper = WrapGeneticGridSearch(self._scored_step(0))
        wrapper.configure({"nb_generations": 2, "nb_estimators": 4})
        initial_generation = [self._scored_step(score) for score in [1, 4, 2, 3]]

        with patch.object(wrapper, "random_generation", side_effect=[
            *initial_generation, self._scored_step(0)
        ]), patch.object(wrapper, "random_mutation", side_effect=deepcopy):
            results = wrapper.run(self._make_candidate())

        self.assertEqual(max(result.get_main_metric_value() for result in results), 4)
        self.assertTrue(all(result.stacked_path == [] for result in results))
        self.assertEqual({result.config_snapshot["score"] for result in results}, {0, 4})

    def test_deduplication_compares_every_configuration_key(self) -> None:
        wrapper = WrapGeneticGridSearch(self._scored_step(0))
        wrapper.configure({"nb_generations": 2, "nb_estimators": 4})
        initial_generation = [self._scored_step(score) for score in [0, 1, 2, 3]]

        with patch.object(wrapper, "random_generation", side_effect=[
            *initial_generation, self._scored_step(0)
        ]), patch.object(wrapper, "random_mutation", side_effect=[
            self._scored_step(1), self._scored_step(2)
        ]):
            results = wrapper.run(self._make_candidate())

        self.assertEqual(len(results), 4)
        self.assertEqual({result.config_snapshot["score"] for result in results}, {0, 1, 2, 3})

    def test_deduplicated_population_of_one_survives_multiple_generations(self) -> None:
        step = ScoredGeneticStep({"score": {"default": 1, "range": [1, 1]}})
        wrapper = WrapGeneticGridSearch(step)
        wrapper.configure({"nb_generations": 3, "nb_estimators": 4})

        results = wrapper.run(self._make_candidate())

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].get_main_metric_value(), 1)

    def test_no_candidates_stops_search_cleanly(self) -> None:
        wrapper = WrapGeneticGridSearch(EmptyGeneticStep())
        wrapper.configure({"nb_generations": 3, "nb_estimators": 3})

        self.assertEqual(wrapper.run(self._make_candidate()), [])

    def test_crashed_workers_do_not_reuse_previous_candidates(self) -> None:
        step = EmptyGeneticStep()
        step.candidate = [self._make_candidate()]
        wrapper = WrapGeneticGridSearch(step)
        wrapper.configure({"nb_generations": 3, "nb_estimators": 2})

        with patch.object(EmptyGeneticStep, "run", side_effect=RuntimeError("test failure")), \
            patch("iaml.worker_manager.Logger.error"):
            results = wrapper.run(self._make_candidate())

        self.assertEqual(results, [])

    def test_multiple_generations_require_evaluation_scores(self) -> None:
        step = DummyGeneticStep({"alpha": {"default": 1, "range": [1, 5]}})
        wrapper = WrapGeneticGridSearch(step)
        wrapper.configure({"nb_generations": 2, "nb_estimators": 2})

        with self.assertRaisesRegex(ValueError, "evaluat|metric|score"):
            wrapper.run(self._make_candidate())

    def test_real_predictor_is_evaluated_across_three_generations(self) -> None:
        values = np.linspace(0.1, 4.1, 30)
        dataset = self.make_dataset(pd.DataFrame({"a": values}), y=2.3 * values + 0.17)
        step = ActDecisionTreeRegressor()
        evaluated = []

        def evaluate(result: Candidate) -> dict:
            scores = result.training_evaluate(dataset)
            evaluated.append(scores)
            return scores

        wrapper = WrapGeneticGridSearch(step, evaluator=evaluate)
        wrapper.configure({"nb_generations": 3, "nb_estimators": 3})

        results = wrapper.run(Candidate(dataset, metrics=[R2ScoreMetric()]))

        self.assertGreaterEqual(len(evaluated), 5)
        self.assertTrue(results)
        self.assertTrue(all(np.isfinite(scores["r2_score"]) for scores in evaluated))
        for result in results:
            self.assertIn("r2_score", result.computed_metrics)
            result.pipeline.fit(dataset.X, dataset.y)
            predictions = result.predict(dataset.X)
            self.assertEqual(len(predictions), len(dataset.X))
            self.assertTrue(np.isfinite(predictions).all())

    def test_pipeline_round_trip_accepts_explicit_evaluator(self) -> None:
        step = ActDecisionTreeRegressor()
        step.configure({"max_depth": 3})
        evaluator = lambda result: {result.main_metric: 0.5}
        wrapper = WrapGeneticGridSearch(step, evaluator=evaluator)
        wrapper.configure({"nb_generations": 3, "nb_estimators": 2})

        restored = WrapGeneticGridSearch.from_pipeline(
            wrapper.json_pipeline(), evaluator=evaluator
        )

        self.assertIs(restored.evaluator, evaluator)
        self.assertEqual(restored.resume_configuration(), wrapper.resume_configuration())
        self.assertIsInstance(restored.step, ActDecisionTreeRegressor)
        self.assertEqual(restored.step.get_config("max_depth"), 3)

    def test_repeated_run_reevaluates_when_callback_context_changes(self) -> None:
        step = DummyGeneticStep({"alpha": {"default": 1, "range": [1, 1]}})
        candidate = self._make_candidate()
        score = 0.25
        evaluated = []

        def evaluate(result: Candidate) -> dict:
            evaluated.append(result)
            return {result.main_metric: score}

        wrapper = WrapGeneticGridSearch(step, evaluator=evaluate)
        wrapper.configure({"nb_generations": 1, "nb_estimators": 1})
        first = wrapper.run(candidate)
        self.assertEqual(first[0].get_main_metric_value(), 0.25)

        score = 0.75
        second = wrapper.run(candidate)

        self.assertEqual(len(evaluated), 2)
        self.assertEqual(second[0].get_main_metric_value(), 0.75)
