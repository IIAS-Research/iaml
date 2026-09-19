"""Tests for ActTfIdf step."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_tf_idf import ActTfIdf


class TestActTfIdf(StepTestCase):
    def make_text_df(self) -> pd.DataFrame:
        titles = [
            "alpha beta a",
            "alpha beta b",
            "alpha beta c",
            "alpha beta d",
            "alpha beta e",
            "alpha beta f",
            "alpha beta g",
            None,
        ]
        notes = [
            "gamma delta a",
            "gamma delta b",
            "gamma delta c",
            "gamma delta d",
            "gamma delta e",
            "gamma delta f",
            "gamma delta g",
            "gamma delta h",
        ]
        counts = list(range(1, 9))
        return pd.DataFrame({"title": titles, "notes": notes, "count": counts})

    def test_transform_creates_prefixed_columns(self) -> None:
        df = self.make_text_df()
        dataset = self.make_dataset(df)
        step = ActTfIdf()
        self.fit_step(step, dataset)

        result = step.transform(dataset.X.copy())

        expected_columns = {
            "count",
            "title_alpha",
            "title_beta",
            "notes_gamma",
            "notes_delta",
        }
        self.assertEqual(set(result.columns), expected_columns)
        self.assertListEqual(result["count"].tolist(), df["count"].tolist())

    def test_transform_handles_missing_text_as_zero(self) -> None:
        df = self.make_text_df()
        step = ActTfIdf()

        result = self.apply_transform(step, df)

        title_cols = [col for col in result.columns if col.startswith("title_")]
        missing_row = result.loc[7, title_cols].to_numpy()
        self.assertTrue(np.allclose(missing_row, 0.0))

    def test_suitable_detects_short_text(self) -> None:
        df = self.make_text_df()
        step = ActTfIdf()

        self.assertTrue(step.suitable(self.make_dataset(df)))

        numeric_df = pd.DataFrame({"count": [1, 2, 3]})
        self.assertFalse(step.suitable(self.make_dataset(numeric_df)))
