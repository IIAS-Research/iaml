"""Tests for MetaPartialExplorerStep."""

from .step_test_case import StepTestCase

from iaml.decorators.all import is_step
from iaml.meta_partial_explorer_step import MetaPartialExplorerStep
from iaml.step import Step
from iaml.void_step import VoidStep


SINGLE_TAG = "unit_meta_partial_single"
MULTI_TAG = "unit_meta_partial_multi"
MISSING_TAG = "unit_meta_partial_missing"


@is_step(SINGLE_TAG)
class SingleTaggedStep(Step):
    """Step used to exercise MetaPartialExplorerStep tag resolution."""
    def __init__(self) -> None:
        pass


@is_step(MULTI_TAG)
class MultiTaggedStepA(Step):
    """First step sharing the same tag."""
    def __init__(self) -> None:
        pass


@is_step(MULTI_TAG)
class MultiTaggedStepB(Step):
    """Second step sharing the same tag."""
    def __init__(self) -> None:
        pass


class TestMetaPartialExplorerStep(StepTestCase):
    def test_initializes_with_void_step_for_tag(self) -> None:
        step = MetaPartialExplorerStep(tag=SINGLE_TAG)

        self.assertTrue(hasattr(step, "steps"))
        self.assertEqual(len(step.steps), 1)
        void_step = step.steps[0]
        self.assertIsInstance(void_step, VoidStep)
        self.assertIsInstance(void_step.step_to_mimic, SingleTaggedStep)
        self.assertEqual(void_step.tags, {SINGLE_TAG})

    def test_uses_single_step_when_multiple_match(self) -> None:
        step = MetaPartialExplorerStep(tag=MULTI_TAG)

        self.assertEqual(len(step.steps), 1)
        void_step = step.steps[0]
        self.assertIsInstance(void_step, VoidStep)
        self.assertIn(type(void_step.step_to_mimic), {MultiTaggedStepA, MultiTaggedStepB})
        self.assertEqual(void_step.tags, {MULTI_TAG})

    def test_missing_tag_leaves_no_steps(self) -> None:
        step = MetaPartialExplorerStep(tag=MISSING_TAG)

        self.assertFalse(getattr(step, "steps", None))
