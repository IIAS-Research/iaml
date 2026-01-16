"""Tests for MetaExplorerStep."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.candidate import Candidate
from iaml.decorators.runner import runner
from iaml.meta_explorer_step import MetaExplorerStep
from iaml.step import Step


class MarkerStep(Step):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        candidate.stacked_path.append(self.label)
        return candidate.to_output()


class TestMetaExplorerStep(StepTestCase):
    def _make_candidate(self) -> Candidate:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        dataset = self.make_dataset(df, y=[0, 1])
        return Candidate(dataset)

    def test_run_collects_outputs_for_each_step(self) -> None:
        candidate = self._make_candidate()
        step = MetaExplorerStep()
        step.add_step(MarkerStep("first"))
        step.add_step(MarkerStep("second"))

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 2)
        stacked_paths = [tuple(output.stacked_path) for output in outputs]
        self.assertCountEqual(stacked_paths, [("first",), ("second",)])

    def test_run_includes_original_candidate_when_requested(self) -> None:
        candidate = self._make_candidate()
        step = MetaExplorerStep(also_explore_without=True)
        step.add_step(MarkerStep("only"))

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 2)
        stacked_paths = [tuple(output.stacked_path) for output in outputs]
        self.assertIn((), stacked_paths)
        self.assertIn(("only",), stacked_paths)

    def test_add_step_marks_interchangeable_and_respects_enable(self) -> None:
        step = MetaExplorerStep()
        step.enable = False
        child = MarkerStep("child")

        step.add_step(child)

        self.assertTrue(child.is_interchangeable)
        self.assertFalse(child.enable)
