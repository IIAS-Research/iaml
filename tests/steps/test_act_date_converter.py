"""Tests for ActDateConverter."""
import pandas as pd

from .step_test_case import StepTestCase
from iaml.actionables.features_precleaning.act_date_converter import ActDateConverter


class TestActDateConverter(StepTestCase):
    def test_converts_dates_when_error_ratio_within_limit(self) -> None:
        values = [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "not_a_date",
            "2020-13-01",
            "32/01/2020",
        ]
        df = pd.DataFrame({"date_text": values})
        step = ActDateConverter()
        step.configure("sample_size", -1)
        step.configure("authorized_error_ratios", 0.5)

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {"date_text": pd.to_datetime(values, errors="coerce")}
        )
        self.assertFrameEqual(result, expected)

    def test_skips_conversion_when_errors_exceed_threshold(self) -> None:
        values = [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2020-01-06",
            "not_a_date",
        ]
        df = pd.DataFrame({"date_text": values})
        step = ActDateConverter()
        step.configure("sample_size", -1)
        step.configure("authorized_error_ratios", 0.0)

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"date_text": values})
        self.assertFrameEqual(result, expected)

    def test_suitable_returns_false_without_short_text(self) -> None:
        df = pd.DataFrame({"num": [1, 2, 3]})
        dataset = self.make_dataset(df)
        step = ActDateConverter()

        self.assertFalse(step.suitable(dataset))
