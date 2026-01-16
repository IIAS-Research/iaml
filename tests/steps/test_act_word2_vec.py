"""Tests for ActWord2Vec step."""
from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pandas as pd

from .step_test_case import StepTestCase


with mock.patch("nltk.download", return_value=True):
    if "iaml.actionables.cleaning.act_word2vec" in sys.modules:
        act_word2vec = importlib.reload(sys.modules["iaml.actionables.cleaning.act_word2vec"])
    else:
        act_word2vec = importlib.import_module("iaml.actionables.cleaning.act_word2vec")

ActWord2Vec = act_word2vec.ActWord2Vec


class TestActWord2Vec(StepTestCase):
    def make_long_text(self, suffix: str = "") -> str:
        return ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 2) + suffix

    def make_long_texts(self, count: int, start: int = 0) -> list[str]:
        return [self.make_long_text(str(i)) for i in range(start, start + count)]

    def make_step(self) -> ActWord2Vec:
        fake_stopwords = mock.Mock()
        fake_stopwords.words.return_value = ["the", "and", "is"]
        with mock.patch.object(act_word2vec, "stopwords", fake_stopwords):
            return ActWord2Vec()

    def test_transform_adds_vector_columns_and_drops_text(self) -> None:
        texts = self.make_long_texts(7)
        df = pd.DataFrame(
            {
                "review": texts,
                "count": list(range(1, len(texts) + 1)),
            }
        )
        step = self.make_step()

        with mock.patch.object(act_word2vec, "word_tokenize", side_effect=lambda text: text.split()):
            result = self.apply_transform(step, df)

        vector_cols = [f"review_vec_{i}" for i in range(100)]
        expected_columns = ["count"] + vector_cols

        self.assertListEqual(result.columns.tolist(), expected_columns)
        self.assertEqual(result.shape[0], df.shape[0])
        non_empty_row = result.loc[0, vector_cols].to_numpy()
        self.assertFalse(np.allclose(non_empty_row, 0.0))

    def test_transform_handles_empty_text_as_zero(self) -> None:
        texts = [self.make_long_text("0"), "", None, np.nan] + self.make_long_texts(3, start=1)
        df = pd.DataFrame(
            {
                "review": texts,
                "count": list(range(1, len(texts) + 1)),
            }
        )
        step = self.make_step()

        with mock.patch.object(act_word2vec, "word_tokenize", side_effect=lambda text: text.split()):
            result = self.apply_transform(step, df)

        vector_cols = [f"review_vec_{i}" for i in range(100)]
        for row_index in [1, 2, 3]:
            empty_row = result.loc[row_index, vector_cols].to_numpy()
            self.assertTrue(np.allclose(empty_row, 0.0))

    def test_suitable_detects_text_columns(self) -> None:
        df = pd.DataFrame({"review": self.make_long_texts(7)})
        step = self.make_step()

        self.assertTrue(step.suitable(self.make_dataset(df)))

        numeric_df = pd.DataFrame({"count": [1, 2, 3]})
        self.assertFalse(step.suitable(self.make_dataset(numeric_df)))
