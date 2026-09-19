"""Tests for StepWrapper."""
import pandas as pd

from .step_test_case import StepTestCase

import iaml
from iaml.candidate import Candidate
from iaml.step import Step
from iaml.step_wrapper import StepWrapper
from iaml.void_step import VoidStep


class DummyStep(Step):
    name = "DummyStep"

    def __init__(self, suitable: bool = True, priority: float = 0.0, count: int = 1) -> None:
        super().__init__()
        self._suitable = suitable
        self._priority = priority
        self._count = count
        self.fit_called = False

    def suitable(self, dataset) -> bool:
        return self._suitable

    def priorize(self, candidate: Candidate = None) -> float:
        return self._priority

    def count_steps(self) -> int:
        return self._count

    def fit(self, dataset) -> "DummyStep":
        self.fit_called = True
        return self


class TestStepWrapper(StepTestCase):
    def test_run_delegates_and_counts(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dataset = self.make_dataset(df, y=[0, 1])
        candidate = Candidate(dataset)
        child = DummyStep(suitable=True, priority=0.7, count=2)
        wrapper = StepWrapper(child)

        result = wrapper.run(candidate)

        self.assertTrue(child.fit_called)
        self.assertTrue(wrapper.suitable(dataset))
        self.assertEqual(wrapper.priorize(candidate), 0.7)
        self.assertEqual(wrapper.count_steps(), 3)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Candidate)

    def test_run_skips_when_child_unsuitable(self) -> None:
        df = pd.DataFrame({"a": [1, 2]})
        dataset = self.make_dataset(df, y=[0, 1])
        candidate = Candidate(dataset)
        child = DummyStep(suitable=False)
        wrapper = StepWrapper(child)

        result = wrapper.run(candidate)

        self.assertFalse(child.fit_called)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], candidate)

    def test_wrap_sets_step_and_rejects_non_step(self) -> None:
        child = DummyStep()
        wrapper = StepWrapper(child)
        new_child = DummyStep()

        wrapper.wrap(new_child)

        self.assertIs(wrapper.step, new_child)
        with self.assertRaises(ValueError):
            wrapper.wrap(object())

    def test_from_pipeline_requires_single_child(self) -> None:
        with self.assertRaises(TypeError):
            StepWrapper.from_pipeline({"step": "StepWrapper"})

    def test_from_pipeline_loads_child_and_json(self) -> None:
        if not hasattr(iaml, "StepWrapper"):
            setattr(iaml, "StepWrapper", StepWrapper)
        if not hasattr(iaml, "VoidStep"):
            setattr(iaml, "VoidStep", VoidStep)
        wrapper = StepWrapper(VoidStep(step_to_mimic=Step()))
        pipeline = wrapper.json_pipeline()

        loaded = StepWrapper.from_pipeline(pipeline)

        self.assertIsInstance(loaded, StepWrapper)
        self.assertIsInstance(loaded.step, VoidStep)
        loaded_pipeline = loaded.json_pipeline()
        self.assertEqual(loaded_pipeline["step"], "StepWrapper")
        self.assertEqual(loaded_pipeline["children"][0]["step"], "VoidStep")
