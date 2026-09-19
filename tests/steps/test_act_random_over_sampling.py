"""Tests for ActRandomOverSampling."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.imbalance.act_random_over_sampling import ActRandomOverSampling


class TestActRandomOverSampling(StepTestCase):
    def test_resample_balances_minority_class(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4],
                "feature_b": [10, 11, 12, 13],
            }
        )
        y = [0, 0, 0, 1]
        step = ActRandomOverSampling()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        counts = pd.Series(y_resampled).value_counts().to_dict()
        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 3)
        self.assertEqual(len(X_resampled), 6)
        self.assertListEqual(list(X_resampled.columns), ["feature_a", "feature_b"])

    def test_resample_preserves_original_rows(self) -> None:
        df = pd.DataFrame(
            {
                "value": [5, 6, 7],
                "score": [0.1, 0.2, 0.3],
            }
        )
        y = [0, 0, 1]
        step = ActRandomOverSampling()

        X_resampled, _ = self.apply_resample(step, df, y)

        original_rows = [tuple(row) for row in df.to_numpy().tolist()]
        resampled_rows = [tuple(row) for row in X_resampled.to_numpy().tolist()]
        for row in original_rows:
            self.assertIn(row, resampled_rows)

    def test_resample_with_string_labels(self) -> None:
        df = pd.DataFrame({"value": [1, 2, 3], "score": [2.0, 4.0, 6.0]})
        y = ["no", "no", "yes"]
        step = ActRandomOverSampling()

        _, y_resampled = self.apply_resample(step, df, y)

        counts = pd.Series(y_resampled).value_counts().to_dict()
        self.assertEqual(counts["no"], 2)
        self.assertEqual(counts["yes"], 2)
