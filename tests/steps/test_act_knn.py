"""Tests for ActKNN step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_knn import ActKNN


class TestActKNN(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5],
                "f2": [1, 0, 1, 0, 1, 0],
                "f3": [2.0, 1.0, 2.5, 1.5, 3.0, 0.5],
            }
        )

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActKNN()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_configuration_passthrough_and_neighbors_cap(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [0, 1, 2],
                "f2": [1, 0, 1],
                "f3": [0.5, 1.5, 2.5],
            }
        )
        y = [0, 1, 0]
        step = ActKNN()
        step.configure("metric", "manhattan")
        step.configure("weights", "distance")
        step.configure("n_neighbors", 10)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertEqual(step.model.metric, "manhattan")
        self.assertEqual(step.model.weights, "distance")
        self.assertEqual(step.model.n_neighbors, len(df))

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame(
            {
                "f1": list(range(10)),
                "f2": [value % 2 for value in range(10)],
            }
        )
        step = ActKNN()

        binary = self.make_dataset(
            df, y=["yes" if value % 2 == 0 else "no" for value in range(10)]
        )
        multilabel = self.make_dataset(
            df,
            y=[
                [value % 3 == 0, value % 3 == 1, value % 3 == 2]
                for value in range(10)
            ],
        )
        continuous = self.make_dataset(
            df, y=[value / 10.0 for value in range(10)]
        )

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))
