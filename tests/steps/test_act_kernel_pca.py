"""Tests for ActKernelPCA step."""
from unittest import mock

import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_kernel_pca import ActKernelPCA


class TestActKernelPCA(StepTestCase):
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
        step = ActKernelPCA()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (len(df), 2))
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_fit_retries_with_gamma_on_value_error(self) -> None:
        df = self._make_numeric_df()
        dataset = self.make_dataset(df)
        step = ActKernelPCA()

        class DummyKernelPCA:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                DummyKernelPCA.instances.append(self)

            def fit(self, X):
                if len(DummyKernelPCA.instances) == 1:
                    raise ValueError("trigger retry")
                return self

        with mock.patch(
            "iaml.actionables.features_preprocessing.act_kernel_pca.KernelPCA",
            DummyKernelPCA,
        ):
            self.fit_step(step, dataset)

        self.assertEqual(len(DummyKernelPCA.instances), 2)
        self.assertNotIn("gamma", DummyKernelPCA.instances[0].kwargs)
        expected_gamma = 1 / df.shape[1] + 0.05
        self.assertAlmostEqual(DummyKernelPCA.instances[1].kwargs["gamma"], expected_gamma)
        self.assertIs(step.preprocessor, DummyKernelPCA.instances[1])
