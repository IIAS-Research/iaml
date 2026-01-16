"""Tests for ActPowerTransformer step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_power_transformer import ActPowerTransformer


class TestActPowerTransformer(StepTestCase):
    def _make_numeric_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [10.0, 11.0, 9.0, 12.0, 8.0, 7.0],
                "f3": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            }
        )

    def test_transform_standardizes_output_by_default(self) -> None:
        df = self._make_numeric_df()
        step = ActPowerTransformer()

        result = self.apply_transform(step, df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, df.shape)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        means = result.mean().to_numpy()
        stds = result.std(ddof=0).to_numpy()
        self.assertTrue(np.allclose(means, np.zeros(df.shape[1]), atol=1e-6))
        self.assertTrue(np.allclose(stds, np.ones(df.shape[1]), atol=1e-6))

    def test_transform_handles_negative_values(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [-1.0, -0.5, 0.0, 0.5],
                "f2": [2.0, -2.0, 3.0, -1.0],
            }
        )
        step = ActPowerTransformer()

        result = self.apply_transform(step, df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, df.shape)
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_suitable_switches_box_cox_on_non_negative_data(self) -> None:
        df = pd.DataFrame({"f1": [0.0, 1.0], "f2": [2.0, 3.0]})
        dataset = self.make_dataset(df)
        step = ActPowerTransformer()
        step.configure("method", "box-cox")

        is_ok = step.suitable(dataset)

        self.assertTrue(is_ok)
        self.assertEqual(step.get_config("method"), "yeo-johnson")
