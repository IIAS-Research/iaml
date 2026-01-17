"""Tests for ActMissingIndicator."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_missing_indicator import ActMissingIndicator


class TestActMissingIndicator(StepTestCase):
    def test_adds_indicators_for_all_columns(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10.0, None, 30.0],
                "score": [1.0, 2.0, None],
                "city": ["paris", None, "london"],
            }
        )
        step = ActMissingIndicator()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "age": [10.0, None, 30.0],
                "score": [1.0, 2.0, None],
                "city": ["paris", None, "london"],
                "age_is_missing": pd.Series([0, 1, 0], dtype="int8"),
                "score_is_missing": pd.Series([0, 0, 1], dtype="int8"),
                "city_is_missing": pd.Series([0, 1, 0], dtype="int8"),
            }
        )
        self.assertFrameEqual(result, expected)

    def test_missing_only_limits_indicators(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10, 11],
                "score": [1.0, None],
                "city": ["paris", "rome"],
            }
        )
        step = ActMissingIndicator()
        step.configure("features", "missing-only")

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "age": [10, 11],
                "score": [1.0, None],
                "city": ["paris", "rome"],
                "score_is_missing": pd.Series([0, 1], dtype="int8"),
            }
        )
        self.assertFrameEqual(result, expected)

    def test_uses_unique_name_when_suffix_collides(self) -> None:
        df = pd.DataFrame(
            {
                "age": [1.0, None],
                "age_is_missing": [9, 9],
            }
        )
        step = ActMissingIndicator()
        step.configure("features", "missing-only")

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "age": [1.0, None],
                "age_is_missing": [9, 9],
                "age_is_missing_1": pd.Series([0, 1], dtype="int8"),
            }
        )
        self.assertFrameEqual(result, expected)
