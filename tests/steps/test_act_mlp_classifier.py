"""Tests for ActMLPClassifier step."""
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_mlp_classifier import (
    ActMLPClassifier,
)


class TestActMLPClassifier(StepTestCase):
    def _make_features(self, n_samples: int = 30) -> pd.DataFrame:
        values = list(range(n_samples))
        return pd.DataFrame(
            {
                "f1": values,
                "f2": [value % 2 for value in values],
                "f3": [value / 10.0 for value in values],
            }
        )

    def _fit_without_warnings(self, step: ActMLPClassifier, dataset) -> ActMLPClassifier:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return self.fit_step(step, dataset)

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [value % 2 for value in range(len(df))]
        step = ActMLPClassifier()
        step.configure("node_per_layer", 16)
        dataset = self.make_dataset(df, y)

        result = self._fit_without_warnings(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_configuration_builds_hidden_layers(self) -> None:
        df = self._make_features()
        y = [value % 2 for value in range(len(df))]
        step = ActMLPClassifier()
        step.configure(
            {
                "hidden_layer_count": 3,
                "node_per_layer": 16,
                "activation": "tanh",
                "alpha": 0.05,
                "learning_rate_init": 0.01,
            }
        )
        dataset = self.make_dataset(df, y)

        self._fit_without_warnings(step, dataset)

        self.assertEqual(list(step.model.hidden_layer_sizes), [16, 16, 16])
        self.assertEqual(step.model.activation, "tanh")
        self.assertAlmostEqual(step.model.alpha, 0.05)
        self.assertAlmostEqual(step.model.learning_rate_init, 0.01)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActMLPClassifier()

        binary = self.make_dataset(df, y=["yes", "no", "yes", "no"])
        multiclass = self.make_dataset(df, y=["a", "b", "c", "a"])
        multilabel = self.make_dataset(
            df,
            y=[[True, False], [False, True], [True, True], [False, False]],
        )
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertTrue(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))
