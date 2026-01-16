"""Tests for ActRBFSampler step."""
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_rbf_sampler import ActRBFSampler


class TestActRBFSampler(StepTestCase):
    def _make_numeric_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [0.5, 1.5, 2.5, 3.5],
                "f3": [2, 3, 4, 5],
            }
        )

    def test_transform_matches_sklearn_with_configured_components(self) -> None:
        df = self._make_numeric_df()
        step = ActRBFSampler()
        step.configure("n_components", 4)

        result = self.apply_transform(step, df)

        expected = RBFSampler(**step.passthrough_parameters()).fit_transform(df)
        expected_df = pd.DataFrame(expected)

        self.assertFrameEqual(result, expected_df)
        self.assertEqual(result.shape, (len(df), 4))
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_transform_is_deterministic_after_fit(self) -> None:
        df = self._make_numeric_df()
        step = ActRBFSampler()
        step.configure("n_components", 3)

        dataset = self.make_dataset(df)
        step.fit(dataset)

        first = step.transform(dataset.X.copy())
        second = step.transform(dataset.X.copy())

        self.assertFrameEqual(first, second)
