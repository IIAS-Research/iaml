"""Tests for MetaOrderedStep."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.candidate import Candidate
from iaml.decorators.runner import runner
from iaml.meta_ordered_step import MetaOrderedStep
from iaml.step import Step


class MarkerStep(Step):
    def __init__(self, label: str, priority: float = 0.0) -> None:
        super().__init__()
        self.label = label
        self.priority = priority

    def priorize(self, candidate: Candidate = None) -> float:
        return self.priority

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        candidate.stacked_path.append(self.label)
        return candidate.to_output()


class BranchStep(Step):
    def __init__(self, labels: list[str]) -> None:
        super().__init__()
        self.labels = labels

    @runner
    def run(self, candidate: Candidate) -> list[Candidate]:
        outputs: list[Candidate] = []
        for label in self.labels:
            output = candidate.to_output()
            output.stacked_path.append(label)
            outputs.append(output)
        return outputs


class TestMetaOrderedStep(StepTestCase):
    def _make_candidate(self) -> Candidate:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dataset = self.make_dataset(df, y=[0, 1])
        return Candidate(dataset)

    def test_run_preserves_list_order(self) -> None:
        candidate = self._make_candidate()
        step = MetaOrderedStep()
        step.add_step(MarkerStep("first", priority=0.5))
        step.add_step(MarkerStep("second", priority=0.9))
        step.add_step(MarkerStep("third", priority=0.1))

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].stacked_path, ["first", "second", "third"])

    def test_run_without_steps_returns_input(self) -> None:
        candidate = self._make_candidate()
        candidate.stacked_path.append("start")
        step = MetaOrderedStep()

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].stacked_path, ["start"])

    def test_run_passes_candidate_list_between_steps(self) -> None:
        candidate = self._make_candidate()
        step = MetaOrderedStep()
        step.add_step(BranchStep(["left", "right"]))
        step.add_step(MarkerStep("tail"))

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 2)
        stacked_paths = [tuple(output.stacked_path) for output in outputs]
        self.assertCountEqual(stacked_paths, [("left", "tail"), ("right", "tail")])
