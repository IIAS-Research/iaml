"""Tests for ActMinMaxScaler."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.normalize.act_minmax_scaler import ActMinMaxScaler


class TestActMinMaxScaler(StepTestCase):
    def test_scales_numeric_columns_only(self) -> None:
        df = pd.DataFrame(
            {
                "age": [1, 2, 3],
                "score": [2.0, 4.0, 6.0],
                "city": ["paris", "lyon", "nice"],
            }
        )
        step = ActMinMaxScaler()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "age": [0.0, 0.5, 1.0],
                "score": [0.0, 0.5, 1.0],
                "city": ["paris", "lyon", "nice"],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_transform_uses_fitted_range(self) -> None:
        train = pd.DataFrame(
            {
                "age": [0, 10],
                "score": [1.0, 3.0],
                "city": ["a", "b"],
            }
        )
        step = ActMinMaxScaler()
        dataset = self.make_dataset(train)
        self.fit_step(step, dataset)

        new = pd.DataFrame(
            {
                "age": [5, 10],
                "score": [2.0, 3.0],
                "city": ["c", "d"],
            }
        )
        result = step.transform(new.copy())

        expected = pd.DataFrame(
            {
                "age": [0.5, 1.0],
                "score": [0.5, 1.0],
                "city": ["c", "d"],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_no_numeric_columns_returns_input(self) -> None:
        df = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        step = ActMinMaxScaler()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        self.assertFrameEqual(result, expected)
