"""Tests for ActMultinomialNB step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_multinomial_nb import (
    ActMultinomialNB,
)


class TestActMultinomialNB(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1, 0, 3, 0, 2, 1],
                "f2": [0, 1, 0, 2, 1, 0],
                "f3": [2, 1, 1, 0, 3, 2],
            }
        )

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActMultinomialNB()
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)

        predictions = np.asarray(step.predict(df)).ravel()
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), set(y))
        self.assertTrue(set(predictions).issubset(set(step.classes_)))

        proba = np.asarray(step.predict_proba(df))
        self.assertEqual(proba.shape, (len(df), len(step.classes_)))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActMultinomialNB()
        step.configure("alpha", 0.25)
        step.configure("fit_prior", False)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertTrue(np.allclose(step.model.alpha, 0.25))
        self.assertFalse(step.model.fit_prior)

    def test_suitable_target_types_and_non_negative(self) -> None:
        df = pd.DataFrame({"f1": list(range(15)), "f2": [1, 0] * 7 + [1]})
        step = ActMultinomialNB()

        binary = self.make_dataset(df, y=[0, 1] * 7 + [0])
        multiclass = self.make_dataset(df, y=["a", "b", "c"] * 5)
        continuous = self.make_dataset(df, y=np.linspace(0.1, 1.5, len(df)))
        negative = self.make_dataset(
            pd.DataFrame({"f1": [-1] + list(range(1, 15)), "f2": [1, 0] * 7 + [1]}),
            y=[0, 1] * 7 + [0],
        )

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertFalse(step.suitable(continuous))
        self.assertFalse(step.suitable(negative))
        with self.assertRaises(ValueError):
            step.fit(negative)
