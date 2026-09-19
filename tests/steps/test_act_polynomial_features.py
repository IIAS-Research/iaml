"""Tests for ActPolynomialFeatures."""
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_polynomial_features import (
    ActPolynomialFeatures,
)


class TestActPolynomialFeatures(StepTestCase):
    def test_default_transform_matches_sklearn(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        step = ActPolynomialFeatures()

        result = self.apply_transform(step, df)

        expected = PolynomialFeatures(**step.passthrough_parameters()).fit_transform(df)
        expected_df = pd.DataFrame(expected)

        self.assertFrameEqual(result, expected_df)

    def test_configured_interactions_without_bias(self) -> None:
        df = pd.DataFrame({"x": [0, 1], "y": [2, 3]})
        step = ActPolynomialFeatures()
        step.configure(
            {
                "include_bias": False,
                "interaction_only": True,
                "degree": 2,
            }
        )

        result = self.apply_transform(step, df)

        expected = PolynomialFeatures(
            include_bias=False,
            interaction_only=True,
            degree=2,
        ).fit_transform(df)
        expected_df = pd.DataFrame(expected)

        self.assertFrameEqual(result, expected_df)

    def test_bias_column_is_all_ones(self) -> None:
        df = pd.DataFrame({"x": [1, 2, 3]})
        step = ActPolynomialFeatures()

        result = self.apply_transform(step, df)

        self.assertEqual(result.shape[1], 4)
        self.assertTrue((result.iloc[:, 0] == 1).all())
