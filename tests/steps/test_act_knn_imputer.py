"""Tests for ActKNNImputer."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_knn_imputer import ActKNNImputer


class TestActKNNImputer(StepTestCase):
    def test_imputes_numeric_and_preserves_non_numeric(self) -> None:
        df = pd.DataFrame(
            {
                "age": [1.0, 2.0, None, 4.0],
                "score": [10.0, None, 30.0, 40.0],
                "city": ["a", "b", "c", "d"],
            }
        )
        step = ActKNNImputer()
        step.configure({"n_neighbors": 1})

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "age": [1.0, 2.0, 4.0, 4.0],
                "score": [10.0, 10.0, 30.0, 40.0],
                "city": ["a", "b", "c", "d"],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_all_nan_numeric_falls_back_to_zero(self) -> None:
        df = pd.DataFrame(
            {"all_nan": [float("nan"), float("nan")]}
        )
        step = ActKNNImputer()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"all_nan": [0.0, 0.0]})
        self.assertFrameEqual(result, expected)
