"""Tests for ActTargetEncoder."""
import unittest

import pandas as pd

from .step_test_case import StepTestCase

try:
    from iaml.actionables.cleaning.act_target_encoder import ActTargetEncoder
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        ActTargetEncoder = None
    else:
        raise

from iaml.data_type import DataType
from iaml.dataset import Dataset


def make_dataset_with_types(
    df: pd.DataFrame, y: list, types: dict[str, DataType]
) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=y, columns_types=columns_types)


class TestActTargetEncoder(StepTestCase):
    @unittest.skipIf(ActTargetEncoder is None, "scikit-learn is required")
    def test_oof_encoding_on_training_data(self) -> None:
        df = pd.DataFrame(
            {
                "cat": ["a", "b", "c", "d"],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        y = [0, 1, 0, 1]
        dataset = make_dataset_with_types(
            df,
            y,
            {"cat": DataType.CATEGORICAL, "value": DataType.NUMERIC},
        )
        step = ActTargetEncoder()
        step.configure("n_splits", 2)
        step.configure("shuffle", True)
        step.configure("smoothing", 1.0)

        self.fit_step(step, dataset)
        result = step.transform(dataset.X)

        prior = sum(y) / len(y)
        expected = pd.DataFrame(
            {
                "cat": [prior, prior, prior, prior],
                "value": [10.0, 20.0, 30.0, 40.0],
            }
        )
        self.assertFrameEqual(result, expected, rtol=1e-6, atol=1e-6)

    @unittest.skipIf(ActTargetEncoder is None, "scikit-learn is required")
    def test_unseen_category_uses_global_mean(self) -> None:
        train_df = pd.DataFrame({"cat": ["a", "b", "b"]})
        y = [1, 0, 0]
        dataset = make_dataset_with_types(
            train_df,
            y,
            {"cat": DataType.CATEGORICAL},
        )
        step = ActTargetEncoder()
        step.configure("n_splits", 2)
        step.configure("shuffle", True)
        step.configure("smoothing", 1.0)

        self.fit_step(step, dataset)

        new_df = pd.DataFrame(
            {
                "cat": ["a", "b", "c"],
                "value": [1.0, 2.0, 3.0],
            }
        )
        result = step.transform(new_df)

        expected = pd.DataFrame(
            {
                "cat": [2.0 / 3.0, 1.0 / 9.0, 1.0 / 3.0],
                "value": [1.0, 2.0, 3.0],
            }
        )
        self.assertFrameEqual(result, expected, rtol=1e-6, atol=1e-6)

    @unittest.skipIf(ActTargetEncoder is None, "scikit-learn is required")
    def test_no_categorical_columns_returns_input(self) -> None:
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        y = [0, 1, 0]
        dataset = make_dataset_with_types(
            df,
            y,
            {"value": DataType.NUMERIC},
        )
        step = ActTargetEncoder()

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        expected = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
        self.assertFrameEqual(result, expected)
