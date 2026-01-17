"""Tests for ActHashingVectorizer."""
import unittest

import pandas as pd

from .step_test_case import StepTestCase

try:
    from iaml.actionables.cleaning.act_hashing_vectorizer import ActHashingVectorizer
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        ActHashingVectorizer = None
    else:
        raise


class TestActHashingVectorizer(StepTestCase):
    def make_long_text(self, suffix: str = "") -> str:
        return (
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda " * 2
        ) + suffix

    def make_long_texts(self, count: int, start: int = 0) -> list[str]:
        return [self.make_long_text(str(i)) for i in range(start, start + count)]

    @unittest.skipIf(ActHashingVectorizer is None, "scikit-learn is required")
    def test_vectorizes_text_column(self) -> None:
        texts = self.make_long_texts(7)
        df = pd.DataFrame(
            {
                "text": texts,
                "value": list(range(1, len(texts) + 1)),
            }
        )
        step = ActHashingVectorizer()
        step.configure("n_features", 8)

        result = self.apply_transform(step, df)

        hash_cols = [f"text_hash_{i}" for i in range(8)]
        self.assertListEqual(result.columns.tolist(), ["value"] + hash_cols)
        self.assertListEqual(result["value"].tolist(), df["value"].tolist())
        self.assertTrue((result.loc[0, hash_cols] > 0).any())

    @unittest.skipIf(ActHashingVectorizer is None, "scikit-learn is required")
    def test_missing_text_becomes_zero_vector(self) -> None:
        texts = self.make_long_texts(7)
        missing_index = 3
        texts[missing_index] = None
        df = pd.DataFrame(
            {
                "text": texts,
                "value": list(range(1, len(texts) + 1)),
            }
        )
        step = ActHashingVectorizer()
        step.configure("n_features", 4)

        result = self.apply_transform(step, df)

        hash_cols = [f"text_hash_{i}" for i in range(4)]
        missing_row = result.loc[missing_index, hash_cols]
        self.assertTrue((missing_row == 0.0).all())

    @unittest.skipIf(ActHashingVectorizer is None, "scikit-learn is required")
    def test_no_text_columns_returns_input(self) -> None:
        df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "flag": [0, 1, 0]})
        step = ActHashingVectorizer()

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
