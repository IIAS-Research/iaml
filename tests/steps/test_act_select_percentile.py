"""Tests for ActSelectPercentile."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_select_percentile import (
    ActSelectPercentile,
)


class TestActSelectPercentile(StepTestCase):
    def test_selects_top_percentile_features(self) -> None:
        df = pd.DataFrame(
            {
                "signal1": [0, 1, 0, 1],
                "signal2": [0, 0, 1, 1],
                "noise1": [1, 1, 1, 1],
                "noise2": [0, 0, 0, 0],
            }
        )
        y = [0, 0, 1, 1]
        step = ActSelectPercentile()
        step.configure("percentile", 50.0)

        result = self.apply_transform(step, df, y)

        expected = pd.DataFrame(df[["signal1", "signal2"]].to_numpy())
        self.assertFrameEqual(result, expected)

    def test_suitable_true_for_non_negative_classification(self) -> None:
        df = pd.DataFrame({"a": list(range(10)), "b": list(range(1, 11))})
        dataset = self.make_dataset(df, y=[0, 1] * 5)
        step = ActSelectPercentile()

        self.assertTrue(step.suitable(dataset))

    def test_suitable_rejects_negative_values(self) -> None:
        df = pd.DataFrame({"a": [0, -1], "b": [1, 2]})
        dataset = self.make_dataset(df, y=[0, 1])
        step = ActSelectPercentile()

        self.assertFalse(step.suitable(dataset))

    def test_suitable_rejects_continuous_target(self) -> None:
        df = pd.DataFrame({"a": [0, 1, 2, 3, 4], "b": [1, 2, 3, 4, 5]})
        dataset = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4, 0.5])
        step = ActSelectPercentile()

        self.assertFalse(step.suitable(dataset))
