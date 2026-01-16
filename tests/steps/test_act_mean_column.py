"""Tests for ActMeanColumn."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_mean_column import ActMeanColumn


class TestActMeanColumn(StepTestCase):
    def test_fills_missing_numeric_columns_with_mean(self) -> None:
        train_df = pd.DataFrame(
            {
                "age": [1.0, None, 3.0, 5.0],
                "score": [1.0, 2.0, None, 3.0],
                "city": ["paris", None, "lyon", "dijon"],
            }
        )
        step = ActMeanColumn()

        self.fit_step(step, self.make_dataset(train_df))

        self.assertEqual(len(step.explanations), 2)
        self.assertTrue(any("`age`" in explanation for explanation in step.explanations))
        self.assertTrue(any("`score`" in explanation for explanation in step.explanations))

        transform_df = pd.DataFrame(
            {
                "age": [10.0, None],
                "score": [None, 5.0],
                "city": ["london", None],
            }
        )
        result = step.transform(transform_df.copy())

        expected = pd.DataFrame(
            {
                "age": [10.0, 3.0],
                "score": [2.0, 5.0],
                "city": ["london", None],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_fills_all_missing_numeric_column_with_zero(self) -> None:
        df = pd.DataFrame(
            {
                "empty": [np.nan, np.nan, np.nan],
                "value": [1.0, 2.0, 3.0],
            }
        )
        dataset = self.make_dataset(df)
        step = ActMeanColumn()

        self.fit_step(step, dataset)

        self.assertEqual(len(step.explanations), 1)
        self.assertTrue("`empty`" in step.explanations[0])

        result = step.transform(dataset.X.copy())

        expected = pd.DataFrame(
            {
                "empty": [0.0, 0.0, 0.0],
                "value": [1.0, 2.0, 3.0],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_no_missing_numeric_columns_noop(self) -> None:
        df = pd.DataFrame(
            {
                "age": [1.0, 2.0, 3.0],
                "score": [0.1, 0.2, 0.3],
            }
        )
        dataset = self.make_dataset(df)
        step = ActMeanColumn()

        self.fit_step(step, dataset)

        self.assertEqual(step.explanations, [])

        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)
