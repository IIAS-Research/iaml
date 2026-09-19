"""Tests for ActSplitDate."""
import pandas as pd
from pandas.api.types import is_integer_dtype

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_split_date import ActSplitDate


class TestActSplitDate(StepTestCase):
    def test_split_date_column_adds_date_parts(self) -> None:
        df = pd.DataFrame(
            {
                "created_at": pd.to_datetime(
                    ["2024-01-01 10:20:30", "2024-01-07 23:59:59"]
                ),
                "score": [1, 2],
            }
        )
        step = ActSplitDate()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "created_at": pd.to_datetime(
                    ["2024-01-01 10:20:30", "2024-01-07 23:59:59"]
                ),
                "score": [1, 2],
                "created_at_weekday": [0, 6],
                "created_at_month": [1, 1],
                "created_at_year": [2024, 2024],
                "created_at_hour": [10, 23],
                "created_at_minute": [20, 59],
                "created_at_second": [30, 59],
            }
        )

        self.assertFrameEqual(result, expected, check_dtype=False)
        for suffix in ("weekday", "month", "year", "hour", "minute", "second"):
            column = f"created_at_{suffix}"
            self.assertTrue(
                is_integer_dtype(result[column]),
                f"{column} should be integer dtype",
            )
        self.assertEqual(step.columns, ["created_at"])
        self.assertEqual(len(step.explanations), 1)
        self.assertIn("`created_at`", step.explanations[0])

    def test_handles_missing_dates_with_minus_one(self) -> None:
        df = pd.DataFrame(
            {
                "event_time": pd.to_datetime(
                    ["2024-03-10 01:02:03", None]
                ),
            }
        )
        step = ActSplitDate()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "event_time": pd.to_datetime(
                    ["2024-03-10 01:02:03", None]
                ),
                "event_time_weekday": [6, -1],
                "event_time_month": [3, -1],
                "event_time_year": [2024, -1],
                "event_time_hour": [1, -1],
                "event_time_minute": [2, -1],
                "event_time_second": [3, -1],
            }
        )

        self.assertFrameEqual(result, expected, check_dtype=False)

    def test_suitable_detects_date_columns(self) -> None:
        date_df = pd.DataFrame(
            {"when": pd.to_datetime(["2024-01-01", "2024-01-02"])}
        )
        non_date_df = pd.DataFrame({"when": ["2024-01-01", "2024-01-02"]})
        step = ActSplitDate()

        self.assertTrue(step.suitable(self.make_dataset(date_df)))
        self.assertFalse(step.suitable(self.make_dataset(non_date_df)))
