"""Tests for ActPCA step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_pca import ActPCA


class TestActPCA(StepTestCase):
    def _make_numeric_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [2.0, 1.0, 0.0, 1.0, 2.0, 3.0],
                "f3": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
                "f4": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            }
        )

    def test_transform_reduces_to_configured_components(self) -> None:
        df = self._make_numeric_df()
        step = ActPCA()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (len(df), 2))
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_default_components_reduce_collinear_data(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [2.0, 4.0, 6.0, 8.0],
                "f3": [3.0, 6.0, 9.0, 12.0],
            }
        )
        step = ActPCA()

        result = self.apply_transform(step, df)

        self.assertEqual(result.shape, (len(df), 1))
