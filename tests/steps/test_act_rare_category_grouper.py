"""Tests for ActRareCategoryGrouper."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_rare_category_grouper import ActRareCategoryGrouper


class TestActRareCategoryGrouper(StepTestCase):
    def test_groups_rare_categories_by_min_count(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["paris", "paris", "paris", "london", "rome"],
                "score": [1, 2, 3, 4, 5],
            }
        )
        step = ActRareCategoryGrouper()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "city": ["paris", "paris", "paris", "other", "other"],
                "score": [1, 2, 3, 4, 5],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_no_change_when_no_rare_categories(self) -> None:
        df = pd.DataFrame(
            {
                "color": ["red", "red", "blue", "blue"],
                "score": [10, 11, 12, 13],
            }
        )
        step = ActRareCategoryGrouper()

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)

    def test_categorical_dtype_keeps_categories(self) -> None:
        base_df = pd.DataFrame({"group": ["a", "a", "b", "c"]})
        step = ActRareCategoryGrouper()

        dataset = self.make_dataset(base_df)
        self.fit_step(step, dataset)
        categorical_df = base_df.copy()
        categorical_df["group"] = categorical_df["group"].astype("category")
        result = step.transform(categorical_df)

        self.assertTrue(isinstance(result["group"].dtype, pd.CategoricalDtype))
        self.assertIn("other", result["group"].cat.categories)
        self.assertListEqual(result["group"].tolist(), ["a", "a", "other", "other"])
