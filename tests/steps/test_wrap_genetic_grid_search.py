"""Tests for WrapGeneticGridSearch."""
import pandas as pd
from unittest.mock import patch

from .step_test_case import StepTestCase

from iaml.candidate import Candidate
from iaml.decorators.all import is_step, runner
from iaml.step import Step
from iaml.wrapper.wrap_genetic_gridsearch import WrapGeneticGridSearch


@is_step("test")
class DummyGeneticStep(Step):
    name = "DummyGeneticStep"

    def __init__(self, configuration: dict) -> None:
        self.configuration = configuration
        self.learning_configuration = self.configuration
        self.run_called = False

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        self.run_called = True
        new_candidate = candidate.to_output()
        new_candidate.config_snapshot = {
            key: item["value"] for key, item in self.learning_configuration.items()
        }
        return new_candidate


class TestWrapGeneticGridSearch(StepTestCase):
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
        snapshot = {
            key: item["value"] for key, item in generated.learning_configuration.items()
        }

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
        self.assertEqual(step.learning_configuration["alpha"]["value"], 3)
        self.assertEqual(mutated.learning_configuration["alpha"]["value"], 4)
        self.assertEqual(mutated.learning_configuration["random_state"]["value"], 11)
        self.assertEqual(mutated.learning_configuration["locked"]["value"], 2.0)

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

        self.assertEqual(mutated_numeric.learning_configuration["alpha"]["value"], 10)
        self.assertIn(mutated_categorical.learning_configuration["mode"]["value"], {"fast", "slow"})

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
