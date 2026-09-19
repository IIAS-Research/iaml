"""Tests for ActKNNRegressor step."""
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

from .step_test_case import StepTestCase

from iaml.actionables.predictors.regressor.act_knn_regressor import (
    ActKNNRegressor,
)


class TestActKNNRegressor(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5],
                "f2": [1, 0, 1, 0, 1, 0],
                "f3": [0.2, 0.4, 0.1, 0.6, 0.3, 0.5],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.5, 1.1, 0.7, 1.4, 0.9, 1.2]
        step = ActKNNRegressor()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, KNeighborsRegressor)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))

    def test_configuration_passthrough_and_neighbors_cap(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [0, 1, 2],
                "f2": [1, 0, 1],
                "f3": [0.1, 0.3, 0.2],
            }
        )
        y = [1.0, 1.5, 2.0]
        step = ActKNNRegressor()
        step.configure("metric", "manhattan")
        step.configure("weights", "distance")
        step.configure("n_neighbors", 10)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertEqual(step.model.metric, "manhattan")
        self.assertEqual(step.model.weights, "distance")
        self.assertEqual(step.model.n_neighbors, len(df))

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActKNNRegressor()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
