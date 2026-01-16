"""Tests for ActBernoulliNb step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_bernoulli_nb import ActBernoulliNb


class TestActBernoulliNb(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 0, 1, 1, 0],
                "f2": [1, 0, 1, 0, 1, 0],
                "f3": [2, 0, 3, 1, 0, 2],
            }
        )

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActBernoulliNb()
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActBernoulliNb()
        step.configure("alpha", 0.25)
        step.configure("fit_prior", False)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertAlmostEqual(step.model.alpha, 0.25)
        self.assertFalse(step.model.fit_prior)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 0, 1], "f2": [1, 1, 0, 0]})
        step = ActBernoulliNb()

        multiclass = self.make_dataset(df, y=["a", "b", "c", "a"])
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])

        self.assertTrue(step.suitable(multiclass))
        self.assertFalse(step.suitable(continuous))
