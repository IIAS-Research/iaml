"""Tests for ActRemoveLowVarianceColumn."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_selection.act_remove_low_variance_column import (
    ActRemoveLowVarianceColumn,
)


class TestActRemoveLowVarianceColumn(StepTestCase):
    def test_drops_low_variance_columns(self) -> None:
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1],
                "signal": [1, 2, 3, 4],
            }
        )
        step = ActRemoveLowVarianceColumn()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"signal": [1, 2, 3, 4]})
        self.assertFrameEqual(result, expected)

    def test_configured_threshold_drops_additional_columns(self) -> None:
        df = pd.DataFrame(
            {
                "constant": [2, 2, 2, 2],
                "low": [0, 0, 1, 1],
                "high": [0, 1, 2, 3],
            }
        )
        step = ActRemoveLowVarianceColumn()
        step.configure("threshold", 0.3)

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"high": [0, 1, 2, 3]})
        self.assertFrameEqual(result, expected)

    def test_keeps_columns_when_variance_is_high(self) -> None:
        df = pd.DataFrame(
            {
                "a": [0, 1, 2, 3],
                "b": [3, 4, 5, 6],
            }
        )
        step = ActRemoveLowVarianceColumn()

        result = self.apply_transform(step, df)

        expected = df.copy()
        self.assertFrameEqual(result, expected)

    def test_suitable_requires_survival_and_low_variance(self) -> None:
        df_low = pd.DataFrame(
            {
                "constant": [1, 1, 1],
                "signal": [1, 2, 3],
            }
        )
        survival_y = [(1, 5), (0, 6), (1, 7)]
        step = ActRemoveLowVarianceColumn()

        self.assertTrue(step.suitable(self.make_dataset(df_low, y=survival_y)))

        df_high = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        self.assertFalse(
            ActRemoveLowVarianceColumn().suitable(self.make_dataset(df_high, y=survival_y))
        )

        self.assertFalse(
            ActRemoveLowVarianceColumn().suitable(self.make_dataset(df_low, y=[0, 1, 0]))
        )
