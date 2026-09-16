"""Tests for the real ActQuantileTransformer component."""
import unittest

import pandas as pd

from iaml.actionables.features_preprocessing.act_quantile_transformer import (
    ActQuantileTransformer,
)
from iaml.candidate import Candidate
from iaml.dataset import Dataset


def _apply_transform(step, df: pd.DataFrame) -> pd.DataFrame:
    step.fit(Dataset(df))
    return step.transform(df.copy())


class TestActQuantileTransformer(unittest.TestCase):
    def test_transform_uniform_on_numeric_columns(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10, 20, 30, 40],
                "score": [1.0, 3.0, 2.0, 5.0],
                "city": ["paris", "lyon", "nice", "dijon"],
            }
        )
        step = ActQuantileTransformer()
        step.configure("output_distribution", "uniform")

        result = _apply_transform(step, df)

        self.assertListEqual(result.columns.tolist(), df.columns.tolist())
        numeric = result[["age", "score"]]
        self.assertGreaterEqual(numeric.min().min(), 0.0)
        self.assertLessEqual(numeric.max().max(), 1.0)
        self.assertListEqual(result["city"].tolist(), df["city"].tolist())
        self.assertNotEqual(numeric.max().max(), df[["age", "score"]].max().max())

    def test_run_builds_a_pipeline_that_reuses_fitted_quantiles(self) -> None:
        train = pd.DataFrame({"age": [0.0, 10.0, 20.0], "city": ["a", "b", "c"]})
        candidate = Candidate(Dataset(train, [0, 1, 0]))
        step = ActQuantileTransformer()
        step.configure("output_distribution", "uniform")

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 1)
        result = outputs[0]
        pd.testing.assert_frame_equal(
            result.dataset.X,
            pd.DataFrame({"age": [0.0, 0.5, 1.0], "city": ["a", "b", "c"]}),
        )
        new = pd.DataFrame({"age": [5.0, 15.0], "city": ["d", "e"]})
        pd.testing.assert_frame_equal(
            result.pipeline.transform(new),
            pd.DataFrame({"age": [0.25, 0.75], "city": ["d", "e"]}),
        )

    def test_no_numeric_columns_returns_input(self) -> None:
        df = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        step = ActQuantileTransformer()

        result = _apply_transform(step, df)

        pd.testing.assert_frame_equal(result, df)

    def test_empty_dataframe_returns_input(self) -> None:
        df = pd.DataFrame(
            {
                "age": pd.Series(dtype="int64"),
                "score": pd.Series(dtype="float64"),
            }
        )
        step = ActQuantileTransformer()

        result = _apply_transform(step, df)

        pd.testing.assert_frame_equal(result, df)
