"""Tests for ActDropHighCardinalityCategorical."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_drop_high_cardinality_categorical import (
    ActDropHighCardinalityCategorical,
)


class TestActDropHighCardinalityCategorical(StepTestCase):
    def test_drops_high_cardinality_category(self) -> None:
        df = pd.DataFrame(
            {
                "high": [
                    "id_0",
                    "id_1",
                    "id_2",
                    "id_3",
                    "id_4",
                    "id_0",
                    "id_1",
                    "id_2",
                    "id_3",
                    "id_4",
                ],
                "low": ["a", "b"] * 5,
                "value": list(range(10)),
            }
        )
        step = ActDropHighCardinalityCategorical()

        result = self.apply_transform(step, df)

        expected = df.drop(columns=["high"])
        self.assertFrameEqual(result, expected)

    def test_hashes_high_cardinality_category(self) -> None:
        df = pd.DataFrame(
            {
                "high": ["a", "b", "c", "d", "a", "b", "c", "d", None, None],
                "low": ["x", "x", "y", "y", "x", "x", "y", "y", "x", "y"],
            }
        )
        step = ActDropHighCardinalityCategorical()
        step.configure(
            {
                "strategy": "hash",
                "hash_bins": 8,
                "min_non_null": 5,
                "missing_value": -1,
            }
        )

        result = self.apply_transform(step, df)

        mask = df["high"].isna()
        self.assertTrue(pd.api.types.is_integer_dtype(result["high"]))
        self.assertTrue((result.loc[mask, "high"] == -1).all())
        self.assertTrue(result.loc[~mask, "high"].between(0, 7).all())
        self.assertFrameEqual(result[["low"]], df[["low"]])

    def test_min_non_null_skips_column(self) -> None:
        df = pd.DataFrame(
            {
                "high": ["a", "b", None, "d"],
                "value": [1, 2, 3, 4],
            }
        )
        step = ActDropHighCardinalityCategorical()
        step.configure({"max_unique": 2, "min_non_null": 5})

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
