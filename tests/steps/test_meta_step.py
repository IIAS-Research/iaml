"""Tests for MetaStep."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.candidate import Candidate
from iaml.decorators.all import is_step
from iaml.decorators.runner import runner
from iaml.metastep import MetaStep
from iaml.step import Step


TAG = "unit_meta_step_tag"


@is_step(TAG)
class TaggedStepA(Step):
    def __init__(self) -> None:
        pass


@is_step(TAG)
class TaggedStepB(Step):
    def __init__(self) -> None:
        pass


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
    def __init__(self, labels: list[str], priority: float = 0.0) -> None:
        super().__init__()
        self.labels = labels
        self.priority = priority

    def priorize(self, candidate: Candidate = None) -> float:
        return self.priority

    @runner
    def run(self, candidate: Candidate) -> list[Candidate]:
        outputs: list[Candidate] = []
        for label in self.labels:
            output = candidate.to_output()
            output.stacked_path.append(label)
            outputs.append(output)
        return outputs


class TestMetaStep(StepTestCase):
    def _make_candidate(self) -> Candidate:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dataset = self.make_dataset(df, y=[0, 1])
        return Candidate(dataset)

    def test_run_prioritizes_steps_by_priorize(self) -> None:
        candidate = self._make_candidate()
        step = MetaStep()
        step.add_step(MarkerStep("low", priority=0.1))
        step.add_step(MarkerStep("high", priority=0.9))

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0].stacked_path, ["high", "low"])

    def test_run_passes_candidate_list_between_steps(self) -> None:
        candidate = self._make_candidate()
        step = MetaStep()
        step.add_step(MarkerStep("tail", priority=0.1))
        step.add_step(BranchStep(["left", "right"], priority=0.9))

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 2)
        stacked_paths = [tuple(output.stacked_path) for output in outputs]
        self.assertCountEqual(stacked_paths, [("left", "tail"), ("right", "tail")])

    def test_add_step_by_tag_adds_matching_steps(self) -> None:
        step = MetaStep(tag=TAG)

        self.assertEqual(len(step.steps), 2)
        step_types = {type(child) for child in step.steps}
        self.assertEqual(step_types, {TaggedStepA, TaggedStepB})
