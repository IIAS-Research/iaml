"""Tests for ActOnehot."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_onehot import ActOnehot
from iaml.data_type import DataType
from iaml.dataset import Dataset


def make_dataset_with_types(df: pd.DataFrame, types: dict[str, DataType]) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=[0] * len(df), columns_types=columns_types)


class TestActOnehot(StepTestCase):
    def test_fit_and_transform_encodes_categorical_columns(self) -> None:
        df = pd.DataFrame(
            {
                "drink": ["coffee", "tea", "coffee"],
                "size": ["S", "M", "L"],
                "price": [1.5, 2.0, 2.5],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "drink": DataType.CATEGORICAL,
                "size": DataType.CATEGORICAL,
                "price": DataType.NUMERIC,
            },
        )
        step = ActOnehot()

        self.fit_step(step, dataset)

        self.assertEqual(step.columns, ["drink", "size"])
        self.assertEqual(len(step.explanations), 2)
        self.assertTrue(any("`drink`" in explanation for explanation in step.explanations))
        self.assertTrue(any("`size`" in explanation for explanation in step.explanations))

        result = step.transform(dataset.X.copy())

        expected = pd.DataFrame(
            {
                "price": [1.5, 2.0, 2.5],
                "drink_coffee": [1.0, 0.0, 1.0],
                "drink_tea": [0.0, 1.0, 0.0],
                "size_L": [0.0, 0.0, 1.0],
                "size_M": [0.0, 1.0, 0.0],
                "size_S": [1.0, 0.0, 0.0],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_transform_ignores_unseen_categories(self) -> None:
        train_df = pd.DataFrame(
            {
                "color": ["red", "blue", "red"],
                "size": ["S", "M", "S"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        train_dataset = make_dataset_with_types(
            train_df,
            {
                "color": DataType.CATEGORICAL,
                "size": DataType.CATEGORICAL,
                "value": DataType.NUMERIC,
            },
        )
        step = ActOnehot()

        self.fit_step(step, train_dataset)

        transform_df = pd.DataFrame(
            {
                "color": ["green", "blue"],
                "size": ["M", "S"],
                "value": [5.0, 6.0],
            },
            index=[10, 20],
        )
        result = step.transform(transform_df.copy())

        expected = pd.DataFrame(
            {
                "value": [5.0, 6.0],
                "color_blue": [0.0, 1.0],
                "color_red": [0.0, 0.0],
                "size_M": [1.0, 0.0],
                "size_S": [0.0, 1.0],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_suitable_detects_categorical_columns(self) -> None:
        df = pd.DataFrame({"city": ["paris", "lyon"], "value": [1.0, 2.0]})
        dataset = make_dataset_with_types(
            df,
            {
                "city": DataType.CATEGORICAL,
                "value": DataType.NUMERIC,
            },
        )
        step = ActOnehot()

        self.assertTrue(step.suitable(dataset))

        non_cat_df = pd.DataFrame(
            {
                "notes": ["alpha", "bravo"],
                "score": [0.1, 0.2],
            }
        )
        non_cat_dataset = make_dataset_with_types(
            non_cat_df,
            {
                "notes": DataType.SHORT_TEXT,
                "score": DataType.NUMERIC,
            },
        )

        self.assertFalse(step.suitable(non_cat_dataset))
