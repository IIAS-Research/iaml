"""Tests for ActRemoveHighCorrelatedColumn."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_selection.act_remove_high_correlated_column import (
    ActRemoveHighCorrelatedColumn,
)


class TestActRemoveHighCorrelatedColumn(StepTestCase):
    def test_fit_drops_highly_correlated_column(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [2, 4, 6, 8],
                "c": [1, 0, 1, 0],
            }
        )
        dataset = self.make_dataset(df)
        step = ActRemoveHighCorrelatedColumn()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.to_drop, ["b"])
        self.assertEqual(len(step.explanations), 1)
        self.assertIn("Dropped column **`b`**", step.explanations[0])
        self.assertIn("**`a`**", step.explanations[0])

        result = step.transform(dataset.X.copy())

        expected = df.drop(["b"], axis=1)
        self.assertFrameEqual(result, expected)

    def test_drops_later_columns_when_multiple_correlated(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [1, 2, 3, 4],
                "c": [1, 2, 3, 4],
            }
        )
        dataset = self.make_dataset(df)
        step = ActRemoveHighCorrelatedColumn()

        self.fit_step(step, dataset)

        self.assertEqual(step.to_drop, ["b", "c"])

        result = step.transform(dataset.X.copy())

        expected = df.drop(["b", "c"], axis=1)
        self.assertFrameEqual(result, expected)

    def test_suitable_false_when_no_high_correlation(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [4, 1, 3, 2],
                "c": [2, 0, 1, 3],
            }
        )
        dataset = self.make_dataset(df)
        step = ActRemoveHighCorrelatedColumn()

        self.assertFalse(step.suitable(dataset))

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
