"""Tests for ActSimpleImputer."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_simple_imputer import ActSimpleImputer


class TestActSimpleImputer(StepTestCase):
    def test_imputes_numeric_and_categorical(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10.0, None, 30.0],
                "score": [1.0, 2.0, None],
                "city": ["paris", None, "paris"],
                "group": ["a", "a", None],
            }
        )
        step = ActSimpleImputer()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "age": [10.0, 20.0, 30.0],
                "score": [1.0, 2.0, 1.5],
                "city": ["paris", "paris", "paris"],
                "group": ["a", "a", "a"],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_all_nan_numeric_falls_back_to_zero(self) -> None:
        df = pd.DataFrame(
            {"age": pd.Series([float("nan"), float("nan")], dtype="float64")}
        )
        step = ActSimpleImputer()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"age": [0.0, 0.0]})
        self.assertFrameEqual(result, expected)

    def test_all_nan_categorical_falls_back_to_empty_string(self) -> None:
        df = pd.DataFrame({"city": [None, None]})
        step = ActSimpleImputer()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"city": ["", ""]})
        self.assertFrameEqual(result, expected)
