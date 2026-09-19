"""Tests for ActOrdinalEncoder."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_ordinal_encoder import ActOrdinalEncoder
from iaml.data_type import DataType
from iaml.dataset import Dataset


def make_dataset_with_types(df: pd.DataFrame, types: dict[str, DataType]) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=[0] * len(df), columns_types=columns_types)


class TestActOrdinalEncoder(StepTestCase):
    def test_fit_and_transform_encodes_sorted_categories(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["b", "a", "c", None],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {"city": DataType.CATEGORICAL, "value": DataType.NUMERIC},
        )
        step = ActOrdinalEncoder()

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        expected = pd.DataFrame(
            {
                "city": [1, 0, 2, -1],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_start_and_unknown_values_are_applied(self) -> None:
        train_df = pd.DataFrame({"fruit": ["banana", "apple", "banana"]})
        dataset = make_dataset_with_types(train_df, {"fruit": DataType.CATEGORICAL})
        step = ActOrdinalEncoder()
        step.configure("start_value", 10)
        step.configure("unknown_value", 99)

        self.fit_step(step, dataset)

        transform_df = pd.DataFrame(
            {
                "fruit": ["banana", "apple", "cherry", None],
                "qty": [1, 2, 3, 4],
            }
        )
        result = step.transform(transform_df.copy())

        expected = pd.DataFrame(
            {
                "fruit": [11, 10, 99, 99],
                "qty": [1, 2, 3, 4],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_sort_categories_false_preserves_order(self) -> None:
        df = pd.DataFrame({"code": ["b", "a", "b"]})
        dataset = make_dataset_with_types(df, {"code": DataType.CATEGORICAL})
        step = ActOrdinalEncoder()
        step.configure("sort_categories", False)

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        expected = pd.DataFrame({"code": [0, 1, 0]})
        self.assertFrameEqual(result, expected)
