"""Tests for ActMissingCountFeature."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_missing_count_feature import ActMissingCountFeature


class TestActMissingCountFeature(StepTestCase):
    def test_adds_missing_count_feature(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10.0, None, 30.0, None],
                "score": [1.0, 2.0, None, None],
                "city": ["paris", "london", "paris", None],
            }
        )
        step = ActMissingCountFeature()

        result = self.apply_transform(step, df)

        expected = df.copy()
        expected["missing_count"] = [0, 1, 1, 3]
        self.assertFrameEqual(result, expected)

    def test_uses_unique_name_when_reserved(self) -> None:
        df = pd.DataFrame(
            {
                "na_count": [5, 0],
                "value": [1.0, None],
                "flag": [None, 2.0],
            }
        )
        step = ActMissingCountFeature()
        step.configure("feature_name", "na_count")

        result = self.apply_transform(step, df)

        expected = df.copy()
        expected["na_count_1"] = [1, 1]
        self.assertFrameEqual(result, expected)
