"""Tests for ActFeatureAgglomeration step."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_feature_agglomeration import (
    ActFeatureAgglomeration,
)


class TestActFeatureAgglomeration(StepTestCase):
    def _make_numeric_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                "f3": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "f4": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )

    def test_transform_reduces_to_configured_clusters(self) -> None:
        df = self._make_numeric_df()
        step = ActFeatureAgglomeration()
        step.configure("n_clusters", 2)

        result = self.apply_transform(step, df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (len(df), 2))

    def test_fit_caps_clusters_to_feature_count(self) -> None:
        df = self._make_numeric_df()[["f1", "f2", "f3"]]
        dataset = self.make_dataset(df)
        step = ActFeatureAgglomeration()

        self.fit_step(step, dataset)

        self.assertEqual(step.get_config("n_clusters"), df.shape[1])
        result = step.transform(dataset.X.copy())
        self.assertEqual(result.shape, (len(df), df.shape[1]))
