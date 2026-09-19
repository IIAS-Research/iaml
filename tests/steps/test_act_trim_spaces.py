"""Example step test for ActTrimSpaces."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.dataset import Dataset
from iaml.actionables.features_precleaning.act_trim_space import ActTrimSpaces


class TestActTrimSpaces(StepTestCase):
    def test_trims_columns_and_values(self) -> None:
        df = pd.DataFrame(
            {
                " name ": [" alice ", "bob"],
                "city": [" paris", "lyon "],
                "age": [1, 2],
            }
        )
        step = ActTrimSpaces()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "name": ["alice", "bob"],
                "city": ["paris", "lyon"],
                "age": [1, 2],
            }
        )

        self.assertFrameEqual(result, expected)

    def test_left_trim_only(self) -> None:
        left_col = " " + "name"
        right_col = "city" + " "
        raw_left = " alice"
        raw_right = "bob "
        raw_both = " " + "paris" + " "
        df = pd.DataFrame(
            {
                left_col: [raw_left, raw_both],
                right_col: [raw_right, raw_both],
            }
        )
        step = ActTrimSpaces()
        step.configure("left_trim", True)
        step.configure("right_trim", False)

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "name": ["alice", "paris "],
                right_col: [raw_right, "paris "],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_right_trim_only(self) -> None:
        right_col = "name" + " "
        left_col = " " + "city"
        raw_left = " alice"
        raw_right = "bob "
        raw_both = " " + "paris" + " "
        df = pd.DataFrame(
            {
                right_col: [raw_right, raw_both],
                left_col: [raw_left, raw_both],
            }
        )
        step = ActTrimSpaces()
        step.configure("left_trim", False)
        step.configure("right_trim", True)

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "name": ["bob", " paris"],
                left_col: [" alice", " paris"],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_handles_nulls_and_non_strings(self) -> None:
        df = pd.DataFrame(
            {
                " name ": [" alice ", None, 3],
                "city": [None, " paris ", 4],
                "age": [1, 2, 3],
            }
        )
        step = ActTrimSpaces()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "name": ["alice", None, 3],
                "city": [None, "paris", 4],
                "age": [1, 2, 3],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_suitable_detects_spaces(self) -> None:
        df = pd.DataFrame({" name ": ["a", "b"]})
        dataset = Dataset(df, y=[0, 1])
        step = ActTrimSpaces()

        self.assertTrue(step.suitable(dataset))

    def test_suitable_returns_false_when_clean(self) -> None:
        df = pd.DataFrame({"name": ["a", "b"]})
        dataset = Dataset(df, y=[0, 1])
        step = ActTrimSpaces()

        self.assertFalse(step.suitable(dataset))
