"""Tests for WrapIterativeGridSearch."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.candidate import Candidate
from iaml.step import Step
from iaml.wrapper.wrap_iterative_gridsearch import WrapIterativeGridSearch


class DummyIterativeStep(Step):
    name = "DummyIterativeStep"

    def __init__(self, configurations: list[dict]) -> None:
        super().__init__()
        self.configuration = configurations
        self.keep_only_first_called = False

    def keep_only_first_config(self) -> None:
        self.keep_only_first_called = True
        if isinstance(self.configuration, list):
            self.configuration = self.configuration[0]

    def run(self, candidate: Candidate) -> list[Candidate]:
        snapshot = {key: item["value"] for key, item in self.configuration.items()}
        new_candidate = candidate.to_output()
        new_candidate.config_snapshot = snapshot
        new_candidate.evaluate = lambda score=1.0: score
        return [new_candidate]


class TestWrapIterativeGridSearch(StepTestCase):
    def _make_candidate(self) -> Candidate:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dataset = self.make_dataset(df, y=[0, 1])
        return Candidate(dataset)

    def test_run_updates_numeric_values_and_keeps_random_state(self) -> None:
        candidate = self._make_candidate()
        configs = [
            {
                "alpha": {"value": 4, "default": 4},
                "random_state": {"value": 7, "default": 7},
            }
        ]
        step = DummyIterativeStep(configs)
        wrapper = WrapIterativeGridSearch(step)
        wrapper.configure("patience", 1)

        results = wrapper.run(candidate)

        self.assertTrue(step.keep_only_first_called)
        self.assertGreater(len(results), 1)
        snapshots = [cand.config_snapshot for cand in results]
        self.assertEqual({snap["random_state"] for snap in snapshots}, {7})
        self.assertGreater(len({snap["alpha"] for snap in snapshots}), 1)

    def test_run_explores_categorical_values(self) -> None:
        candidate = self._make_candidate()
        configs = [
            {
                "mode": {
                    "value": "fast",
                    "default": "fast",
                    "categorical": ["fast", "slow"],
                }
            }
        ]
        step = DummyIterativeStep(configs)
        wrapper = WrapIterativeGridSearch(step)
        wrapper.configure("patience", 1)

        results = wrapper.run(candidate)

        snapshots = [cand.config_snapshot for cand in results]
        self.assertEqual({snap["mode"] for snap in snapshots}, {"fast", "slow"})

    def test_run_with_only_random_state_preserves_value(self) -> None:
        candidate = self._make_candidate()
        configs = [{"random_state": {"value": 3, "default": 3}}]
        step = DummyIterativeStep(configs)
        wrapper = WrapIterativeGridSearch(step)
        wrapper.configure("patience", 1)

        results = wrapper.run(candidate)

        self.assertGreaterEqual(len(results), 1)
        snapshots = [cand.config_snapshot for cand in results]
        self.assertTrue(all("random_state" in snap for snap in snapshots))
        self.assertIn(3, {snap["random_state"] for snap in snapshots})
