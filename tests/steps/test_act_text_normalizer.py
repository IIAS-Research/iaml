"""Tests for ActTextNormalizer."""
import unittest

import pandas as pd

from .step_test_case import StepTestCase

try:
    from iaml.actionables.cleaning.act_text_normalizer import ActTextNormalizer
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        ActTextNormalizer = None
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


class TestActTextNormalizer(StepTestCase):
    @unittest.skipIf(ActTextNormalizer is None, "scikit-learn is required")
    def test_normalizes_text_column(self) -> None:
        df = pd.DataFrame(
            {
                "text": [
                    "Hello, WORLD!!! and the fox",
                    "Tabs\tand  spaces",
                    "Numbers 123 and commas, periods.",
                    "Multiple---dashes and__underscores",
                ],
                "value": [1, 2, 3, 4],
            }
        )
        dataset = make_dataset_with_types(
            df,
            y=[0, 0, 0, 0],
            types={"text": DataType.SHORT_TEXT, "value": DataType.NUMERIC},
        )
        step = ActTextNormalizer()
        step.configure({"stopwords": ["and", "the"]})

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        expected = pd.DataFrame(
            {
                "text": [
                    "hello world fox",
                    "tabs spaces",
                    "numbers 123 commas periods",
                    "multiple dashes underscores",
                ],
                "value": [1, 2, 3, 4],
            }
        )
        self.assertFrameEqual(result, expected)

    @unittest.skipIf(ActTextNormalizer is None, "scikit-learn is required")
    def test_no_text_columns_returns_input(self) -> None:
        df = pd.DataFrame(
            {"category": ["a", "a", "b", "b"], "value": [1.0, 2.0, 3.0, 4.0]}
        )
        dataset = make_dataset_with_types(
            df,
            y=[0, 0, 0, 0],
            types={"category": DataType.CATEGORICAL, "value": DataType.NUMERIC},
        )
        step = ActTextNormalizer()

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)

    @unittest.skipIf(ActTextNormalizer is None, "scikit-learn is required")
    def test_all_operations_disabled_returns_input(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["Hello, WORLD!!!", "and the", "Tabs\tand spaces"],
                "value": [1, 2, 3],
            }
        )
        dataset = make_dataset_with_types(
            df,
            y=[0, 0, 0],
            types={"text": DataType.SHORT_TEXT, "value": DataType.NUMERIC},
        )
        step = ActTextNormalizer()
        step.configure(
            {
                "lowercase": False,
                "remove_punctuation": False,
                "remove_stopwords": False,
                "collapse_whitespace": False,
            }
        )

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)
