"""Tests for ActFrequencyEncoder."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_frequency_encoder import ActFrequencyEncoder
from iaml.data_type import DataType
from iaml.dataset import Dataset


def make_dataset_with_types(df: pd.DataFrame, types: dict[str, DataType]) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=[0] * len(df), columns_types=columns_types)


class TestActFrequencyEncoder(StepTestCase):
    def test_fit_and_transform_encodes_categorical_columns(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["paris", "paris", "lyon", "paris"],
                "size": ["S", "M", "S", "S"],
                "price": [1.0, 2.0, 3.0, 4.0],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "city": DataType.CATEGORICAL,
                "size": DataType.CATEGORICAL,
                "price": DataType.NUMERIC,
            },
        )
        step = ActFrequencyEncoder()

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        expected = pd.DataFrame(
            {
                "city": [0.75, 0.75, 0.25, 0.75],
                "size": [0.75, 0.25, 0.75, 0.75],
                "price": [1.0, 2.0, 3.0, 4.0],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_missing_and_unseen_use_unknown_value(self) -> None:
        train_df = pd.DataFrame({"color": ["red", None, "blue"]})
        dataset = make_dataset_with_types(
            train_df,
            {"color": DataType.CATEGORICAL},
        )
        step = ActFrequencyEncoder()
        step.configure("unknown_value", 0.2)

        self.fit_step(step, dataset)

        transform_df = pd.DataFrame(
            {
                "color": ["red", "green", None],
                "score": [10, 11, 12],
            }
        )
        result = step.transform(transform_df.copy())

        expected = pd.DataFrame(
            {
                "color": [0.5, 0.2, 0.2],
                "score": [10, 11, 12],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_category_dtype_columns_are_encoded(self) -> None:
        df = pd.DataFrame(
            {
                "segment": pd.Series(["a", "a", "b", "a"], dtype="category"),
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "segment": DataType.CATEGORICAL,
                "value": DataType.NUMERIC,
            },
        )
        step = ActFrequencyEncoder()

        self.fit_step(step, dataset)
        transform_df = dataset.X.copy()
        transform_df["segment"] = transform_df["segment"].astype(object)
        result = step.transform(transform_df)

        expected = pd.DataFrame(
            {
                "segment": [0.75, 0.75, 0.25, 0.75],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        self.assertFrameEqual(result, expected)
