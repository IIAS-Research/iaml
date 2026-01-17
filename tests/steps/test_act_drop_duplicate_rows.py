"""Tests for ActDropDuplicateRows."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_precleaning.act_drop_duplicate_rows import (
    ActDropDuplicateRows,
)


class TestActDropDuplicateRows(StepTestCase):
    def test_resample_drops_duplicates_and_aligns_y(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 1, 3],
                "b": ["x", "y", "x", "z"],
            }
        )
        y = [10, 20, 30, 40]
        step = ActDropDuplicateRows()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        expected_X = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )
        self.assertFrameEqual(X_resampled, expected_X)
        self.assertListEqual(list(y_resampled), [10, 20, 40])

    def test_resample_no_duplicates_passthrough(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            }
        )
        y = [5, 6, 7]
        step = ActDropDuplicateRows()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        self.assertFrameEqual(X_resampled, df)
        self.assertListEqual(list(y_resampled), y)

    def test_resample_misaligned_y_returns_unmodified_y(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 1, 2],
                "b": ["x", "x", "y"],
            }
        )
        y = [100, 200]
        step = ActDropDuplicateRows()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        expected_X = pd.DataFrame(
            {
                "a": [1, 2],
                "b": ["x", "y"],
            }
        )
        self.assertFrameEqual(X_resampled, expected_X)
        self.assertListEqual(list(y_resampled), y)

    def test_fit_reports_duplicate_metrics(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 1, 3],
                "b": ["x", "y", "x", "z"],
            }
        )
        dataset = self.make_dataset(df, y=[0, 1, 2, 3])
        step = ActDropDuplicateRows()

        step.fit(dataset)

        self.assertEqual(step.duplicate_count, 1)
        self.assertAlmostEqual(step.duplicate_ratio, 0.25)
        self.assertEqual(len(step.explanations), 1)
        self.assertIn("keep=first", step.explanations[0])
