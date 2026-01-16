"""Tests for ActExtraTreesClassifier step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_extra_trees_classifier import (
    ActExtraTreesClassifier,
)


class TestActExtraTreesClassifier(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0],
                "f3": [5, 4, 3, 2, 1, 0, 1, 2],
            }
        )

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        step = ActExtraTreesClassifier()
        step.configure("n_estimators", 10)
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

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        step = ActExtraTreesClassifier()
        step.configure(
            {
                "max_depth": 3,
                "min_samples_leaf": 2,
                "min_samples_split": 3,
                "bootstrap": True,
                "max_features": "log2",
                "criterion": "entropy",
                "n_estimators": 5,
                "random_state": 7,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertEqual(step.model.max_depth, 3)
        self.assertEqual(step.model.min_samples_leaf, 2)
        self.assertEqual(step.model.min_samples_split, 3)
        self.assertTrue(step.model.bootstrap)
        self.assertEqual(step.model.max_features, "log2")
        self.assertEqual(step.model.criterion, "entropy")
        self.assertEqual(step.model.n_estimators, 5)
        self.assertEqual(step.model.random_state, 7)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActExtraTreesClassifier()

        multiclass = self.make_dataset(df, y=["a", "b", "c", "a"])
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])

        self.assertTrue(step.suitable(multiclass))
        self.assertFalse(step.suitable(continuous))
