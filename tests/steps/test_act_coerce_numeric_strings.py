"""Tests for ActCoerceNumericStrings."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_precleaning.act_coerce_numeric_strings import (
    ActCoerceNumericStrings,
)


class TestActCoerceNumericStrings(StepTestCase):
    def test_coerces_dot_decimal_and_percent(self) -> None:
        df = pd.DataFrame({"amount": ["1,234.50", "2 000", "50%"]})
        step = ActCoerceNumericStrings()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"amount": [1234.5, 2000.0, 0.5]})
        self.assertFrameEqual(result, expected)

    def test_coerces_comma_decimal(self) -> None:
        df = pd.DataFrame({"amount": ["1.234,50", "2.000,25", "300,0"]})
        step = ActCoerceNumericStrings()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"amount": [1234.5, 2000.25, 300.0]})
        self.assertFrameEqual(result, expected)

    def test_skips_columns_with_too_many_errors(self) -> None:
        df = pd.DataFrame({"mixed": ["alpha", "bravo", "1"]})
        step = ActCoerceNumericStrings()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"mixed": ["alpha", "bravo", "1"]})
        self.assertFrameEqual(result, expected)
