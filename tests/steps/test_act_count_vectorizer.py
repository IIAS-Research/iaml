"""Tests for ActCountVectorizer."""
import unittest

import pandas as pd

from .step_test_case import StepTestCase

try:
    from iaml.actionables.cleaning.act_count_vectorizer import ActCountVectorizer
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        ActCountVectorizer = None
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


class TestActCountVectorizer(StepTestCase):
    @unittest.skipIf(ActCountVectorizer is None, "scikit-learn is required")
    def test_vectorizes_short_text_column(self) -> None:
        df = pd.DataFrame(
            {
                "text": [
                    "red blue",
                    "blue green",
                    "red",
                    "green",
                    "red green",
                    "blue",
                    "green blue",
                ],
                "value": [1, 2, 3, 4, 5, 6, 7],
            }
        )
        step = ActCountVectorizer()

        result = self.apply_transform(step, df)

        self.assertNotIn("text", result.columns)
        self.assertIn("value", result.columns)
        self.assertIn("text_red", result.columns)
        self.assertIn("text_blue", result.columns)
        self.assertIn("text_green", result.columns)
        self.assertIn("text_red blue", result.columns)

        row0 = result.iloc[0]
        self.assertEqual(row0["text_red"], 1)
        self.assertEqual(row0["text_blue"], 1)
        self.assertEqual(row0["text_red blue"], 1)
        self.assertEqual(row0["text_green"], 0)

    @unittest.skipIf(ActCountVectorizer is None, "scikit-learn is required")
    def test_no_short_text_columns_returns_input(self) -> None:
        df = pd.DataFrame(
            {
                "category": ["a", "a", "b", "b", "a", "b", "a"],
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )
        step = ActCountVectorizer()

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)

    @unittest.skipIf(ActCountVectorizer is None, "scikit-learn is required")
    def test_empty_vocabulary_returns_input(self) -> None:
        df = pd.DataFrame({"text": ["", "", ""]})
        dataset = make_dataset_with_types(
            df,
            y=[0, 0, 0],
            types={"text": DataType.SHORT_TEXT},
        )
        step = ActCountVectorizer()

        with self.assertRaises(RuntimeError):
            self.fit_step(step, dataset)
