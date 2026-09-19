"""Tests for ActNormalizeColumnNames."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_precleaning.act_normalize_column_names import (
    ActNormalizeColumnNames,
)


class TestActNormalizeColumnNames(StepTestCase):
    def test_normalizes_common_patterns(self) -> None:
        df = pd.DataFrame(
            {
                " Name ": [1, 2],
                "City Name": [3, 4],
                "ZIP-code": [5, 6],
                "___state__": [7, 8],
            }
        )
        step = ActNormalizeColumnNames()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "name": [1, 2],
                "city_name": [3, 4],
                "zip_code": [5, 6],
                "state": [7, 8],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_deduplicates_and_uses_fallback(self) -> None:
        data = [
            [1, 2, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ]
        columns = ["A", "a", "A ", None, "", "__"]
        df = pd.DataFrame(data, columns=columns)
        step = ActNormalizeColumnNames()

        result = self.apply_transform(step, df)

        expected_columns = ["a", "a_1", "a_2", "column", "column_1", "column_2"]
        expected = pd.DataFrame(data, columns=expected_columns)
        self.assertFrameEqual(result, expected)

    def test_suitable_detects_dirty_columns(self) -> None:
        step = ActNormalizeColumnNames()

        dirty = self.make_dataset(pd.DataFrame({"A B": [1], "clean": [2]}))
        clean = self.make_dataset(pd.DataFrame({"a_b": [1], "clean": [2]}))

        self.assertTrue(step.suitable(dirty))
        self.assertFalse(step.suitable(clean))
