"""Tests for ActDropTextualColumn."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_drop_textual_column import ActDropTextualColumn
from iaml.data_type import DataType
from iaml.dataset import Dataset


def make_dataset_with_types(df: pd.DataFrame, types: dict[str, DataType]) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=[0] * len(df), columns_types=columns_types)


class TestActDropTextualColumn(StepTestCase):
    def test_fit_drops_text_columns_and_sets_explanations(self) -> None:
        long_text = "x" * 90
        df = pd.DataFrame(
            {
                "short_text": ["alpha", "bravo"],
                "long_text": [long_text, long_text + "y"],
                "value": [1, 2],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "short_text": DataType.SHORT_TEXT,
                "long_text": DataType.TEXT,
                "value": DataType.NUMERIC,
            },
        )
        step = ActDropTextualColumn()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        expected_dropped = ["short_text", "long_text"]
        expected_explanations = [
            f"Dropped column **`{column}`**." for column in expected_dropped
        ]
        self.assertCountEqual(step.columns_to_drop, expected_dropped)
        self.assertEqual(len(step.columns_to_drop), len(expected_dropped))
        self.assertCountEqual(step.explanations, expected_explanations)
        self.assertEqual(len(step.explanations), len(expected_dropped))

        result = step.transform(dataset.X.copy())

        expected = df.drop(expected_dropped, axis=1)
        self.assertFrameEqual(result, expected)

    def test_no_text_columns_noop_and_not_suitable(self) -> None:
        df = pd.DataFrame(
            {
                "age": [1, 2],
                "score": [0.1, 0.2],
            }
        )
        dataset = self.make_dataset(df)
        step = ActDropTextualColumn()

        self.assertFalse(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.columns_to_drop, [])
        self.assertEqual(step.explanations, [])

        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)
