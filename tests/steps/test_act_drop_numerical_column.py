"""Tests for ActDropNumericalColumn."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_drop_numerical_column import ActDropNumericalColumn


class TestActDropNumericalColumn(StepTestCase):
    def test_fit_drops_numeric_columns_at_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "score": [1.0, None, 3.0, None],
                "ratio": [0.1, 0.2, None, 0.4],
                "name": ["a", None, "c", "d"],
            }
        )
        dataset = self.make_dataset(df)
        step = ActDropNumericalColumn()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.columns_to_drop, ["score"])
        self.assertEqual(len(step.explanations), 1)
        self.assertIn("Dropped column **`score`**", step.explanations[0])
        self.assertIn("50.00%", step.explanations[0])

        result = step.transform(dataset.X.copy())

        expected = df.drop(["score"], axis=1)
        self.assertFrameEqual(result, expected)

    def test_no_numeric_columns_above_threshold_noop(self) -> None:
        df = pd.DataFrame(
            {
                "score": [1.0, None, 3.0, 4.0],
                "ratio": [0.1, 0.2, 0.3, 0.4],
                "name": ["a", "b", "c", "d"],
            }
        )
        dataset = self.make_dataset(df)
        step = ActDropNumericalColumn()

        self.assertFalse(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.columns_to_drop, [])
        self.assertEqual(step.explanations, [])

        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)

    def test_configured_threshold_drops_expected_columns(self) -> None:
        df = pd.DataFrame(
            {
                "score": [1.0, None, 3.0, 4.0],
                "ratio": [10.0, 11.0, 12.0, 13.0],
                "name": ["a", "b", "c", "d"],
            }
        )
        dataset = self.make_dataset(df)
        step = ActDropNumericalColumn()
        step.configure("empty_threshold", 0.25)

        self.assertTrue(step.suitable(dataset))

        result = self.apply_transform(step, df)

        expected = df.drop(["score"], axis=1)
        self.assertEqual(step.columns_to_drop, ["score"])
        self.assertFrameEqual(result, expected)
