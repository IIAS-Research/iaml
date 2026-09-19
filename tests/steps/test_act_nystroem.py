"""Tests for ActNystroem step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_nystroem import ActNystroem


class TestActNystroem(StepTestCase):
    def _make_numeric_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "f3": [0.0, 1.0, 0.5, 1.5, 2.0, 2.5],
            }
        )

    def test_transform_returns_configured_components(self) -> None:
        df = self._make_numeric_df()
        step = ActNystroem()
        step.configure("n_components", 3)

        result = self.apply_transform(step, df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (len(df), 3))
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_suitable_switches_chi2_when_no_negative_values(self) -> None:
        df = pd.DataFrame({"f1": [0.0, 1.0], "f2": [2.0, 3.0]})
        dataset = self.make_dataset(df)
        step = ActNystroem()
        step.configure("kernel", "chi2")

        is_ok = step.suitable(dataset)

        self.assertTrue(is_ok)
        self.assertEqual(step.get_config("kernel"), "rbf")

    def test_suitable_keeps_chi2_when_negative_values_present(self) -> None:
        df = pd.DataFrame({"f1": [-1.0, 1.0], "f2": [0.0, 2.0]})
        dataset = self.make_dataset(df)
        step = ActNystroem()
        step.configure("kernel", "chi2")

        is_ok = step.suitable(dataset)

        self.assertTrue(is_ok)
        self.assertEqual(step.get_config("kernel"), "chi2")
