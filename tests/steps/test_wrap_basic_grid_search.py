"""Tests for WrapBasicGridSearch."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.candidate import Candidate
from iaml.decorators.all import is_step
from iaml.step import Step
from iaml.wrapper.wrap_basic_gridsearch import WrapBasicGridSearch


@is_step("test")
class DummyGridStep(Step):
    name = "DummyGridStep"

    def __init__(self, configuration: dict) -> None:
        self.configuration = configuration

    def run(self, candidate: Candidate) -> Candidate:
        if isinstance(candidate, list):
            candidate = candidate[0]
        snapshot = {key: item["value"] for key, item in self.configuration.items()}
        new_candidate = candidate.to_output()
        new_candidate.config_snapshot = snapshot
        return new_candidate


class TestWrapBasicGridSearch(StepTestCase):
    def test_expands_numeric_and_categorical_values(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        candidate = Candidate(self.make_dataset(df, y=[0, 1]))
        config = {
            "alpha": {"value": 10, "default": 10},
            "mode": {"value": "fast", "default": "fast", "categorical": ["fast", "slow"]},
            "label": {"value": "keep", "default": "keep"},
        }
        step = DummyGridStep(config)
        wrapper = WrapBasicGridSearch(step)

        results = wrapper.run(candidate)

        snapshots = [cand.config_snapshot for cand in results]
        self.assertEqual(
            {snap["alpha"] for snap in snapshots},
            {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        )
        self.assertEqual({snap["mode"] for snap in snapshots}, {"fast", "slow"})
        self.assertEqual({snap["label"] for snap in snapshots}, {"keep"})

    def test_expands_boolean_and_ignores_random_state(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        candidate = Candidate(self.make_dataset(df, y=[0, 1]))
        config = {
            "use_bias": {"value": True, "default": True},
            "random_state": {"value": 123, "default": 123},
            "name": {"value": "base", "default": "base"},
        }
        step = DummyGridStep(config)
        wrapper = WrapBasicGridSearch(step)

        results = wrapper.run(candidate)

        self.assertEqual(len(results), 2)
        snapshots = [cand.config_snapshot for cand in results]
        self.assertEqual({snap["use_bias"] for snap in snapshots}, {True, False})
        self.assertEqual({snap["random_state"] for snap in snapshots}, {123})
        self.assertEqual({snap["name"] for snap in snapshots}, {"base"})
