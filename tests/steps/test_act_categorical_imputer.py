"""Tests for ActCategoricalImputer."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_categorical_imputer import ActCategoricalImputer


class TestActCategoricalImputer(StepTestCase):
    def test_imputes_most_frequent_category(self) -> None:
        values = ["red", None, "red", "blue"]
        fit_df = pd.DataFrame({"color": values})
        step = ActCategoricalImputer()

        dataset = self.make_dataset(fit_df)
        self.fit_step(step, dataset)
        transform_df = pd.DataFrame(
            {"color": pd.Series(values, dtype="category")}
        )
        result = step.transform(transform_df.copy())

        expected = pd.DataFrame(
            {"color": pd.Series(["red", "red", "red", "blue"], dtype="category")}
        )
        self.assertFrameEqual(result, expected)

    def test_missing_strategy_uses_label_and_adds_category(self) -> None:
        values = ["a", None, "b"]
        fit_df = pd.DataFrame({"status": values})
        step = ActCategoricalImputer()
        step.configure({"strategy": "missing", "missing_label": "unknown"})

        dataset = self.make_dataset(fit_df)
        self.fit_step(step, dataset)
        transform_df = pd.DataFrame(
            {"status": pd.Series(values, dtype="category")}
        )
        result = step.transform(transform_df.copy())

        expected = pd.DataFrame(
            {"status": pd.Series(["a", "unknown", "b"], dtype="category")}
        )
        self.assertFrameEqual(result, expected)
        self.assertIn("unknown", result["status"].cat.categories)

    def test_all_missing_falls_back_to_default_label(self) -> None:
        values = [None, None]
        fit_df = pd.DataFrame({"group": values})
        step = ActCategoricalImputer()

        dataset = self.make_dataset(fit_df)
        self.fit_step(step, dataset)
        transform_df = pd.DataFrame(
            {"group": pd.Series(values, dtype="category")}
        )
        result = step.transform(transform_df.copy())

        expected = pd.DataFrame(
            {"group": pd.Series(["missing", "missing"], dtype="category")}
        )
        self.assertFrameEqual(result, expected)
        self.assertIn("missing", result["group"].cat.categories)
