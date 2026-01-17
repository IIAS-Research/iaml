"""Tests for ActDropHighMissingColumns."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_precleaning.act_drop_high_missing_columns import (
    ActDropHighMissingColumns,
)


class TestActDropHighMissingColumns(StepTestCase):
    def test_drops_columns_at_or_above_default_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "half_missing": [1.0, None, 3.0, None],
                "mostly_missing": [None, None, 1.0, None],
                "low_missing": [1.0, 2.0, None, 4.0],
                "clean": ["a", "b", "c", "d"],
            }
        )
        step = ActDropHighMissingColumns()

        self.assertTrue(step.suitable(self.make_dataset(df)))

        result = self.apply_transform(step, df)

        expected = df.drop(columns=["half_missing", "mostly_missing"])
        self.assertFrameEqual(result, expected)

    def test_below_threshold_noop_and_not_suitable(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1.0, None, 3.0, 4.0],
                "b": [10, 11, 12, 13],
            }
        )
        step = ActDropHighMissingColumns()

        self.assertFalse(step.suitable(self.make_dataset(df)))

        result = self.apply_transform(step, df)

        expected = df.copy()
        self.assertFrameEqual(result, expected)

    def test_configured_threshold_one_drops_only_fully_missing(self) -> None:
        df = pd.DataFrame(
            {
                "all_missing": [None, None, None],
                "some_missing": [1.0, None, 3.0],
                "clean": [1, 2, 3],
            }
        )
        step = ActDropHighMissingColumns()
        step.configure("missing_threshold", 1.0)

        result = self.apply_transform(step, df)

        expected = df.drop(columns=["all_missing"])
        self.assertFrameEqual(result, expected)
