"""Tests for ActDropCategoricalColumn."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_drop_categorical_column import ActDropCategoricalColumn
from iaml.data_type import DataType
from iaml.dataset import Dataset


def make_dataset_with_types(df: pd.DataFrame, types: dict[str, DataType]) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=[0] * len(df), columns_types=columns_types)


class TestActDropCategoricalColumn(StepTestCase):
    def test_fit_drops_categorical_and_category_dtype_columns(self) -> None:
        df = pd.DataFrame(
            {
                "cat_obj": ["a", "b", "a", "b"],
                "cat_dtype": pd.Series(["x", "y", "x", "y"], dtype="category"),
                "text": ["alpha", "bravo", "charlie", "delta"],
                "value": [1, 2, 3, 4],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "cat_obj": DataType.CATEGORICAL,
                "cat_dtype": DataType.SHORT_TEXT,
                "text": DataType.SHORT_TEXT,
                "value": DataType.NUMERIC,
            },
        )
        step = ActDropCategoricalColumn()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        expected_dropped = ["cat_obj", "cat_dtype"]
        expected_explanations = [
            f"Dropped column **`{column}`**." for column in expected_dropped
        ]
        self.assertCountEqual(step.columns_to_drop, expected_dropped)
        self.assertCountEqual(step.explanations, expected_explanations)
        self.assertEqual(len(step.explanations), len(step.columns_to_drop))

        result = step.transform(dataset.X.copy())

        expected = df.drop(expected_dropped, axis=1)
        self.assertFrameEqual(result, expected)

    def test_no_categorical_columns_noop_and_not_suitable(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["alpha", "bravo"],
                "value": [1, 2],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "text": DataType.SHORT_TEXT,
                "value": DataType.NUMERIC,
            },
        )
        step = ActDropCategoricalColumn()

        self.assertFalse(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.columns_to_drop, [])
        self.assertEqual(step.explanations, [])

        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)
