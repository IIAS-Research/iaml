"""Tests for ActXGBoost step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_xgboost import ActXGBoost


class TestActXGBoost(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0, 1],
                "f3": [0.2, 0.4, 0.1, 0.3, 0.5, 0.7, 0.6, 0.8, 0.9],
            }
        )

    def _configure_fast(self, step: ActXGBoost) -> None:
        step.configure({"n_estimators": 5, "max_depth": 2, "random_state": 7})

    def test_fit_binary_trains_and_sets_loss_choices(self) -> None:
        df = self._make_features().iloc[:6].copy()
        y = ["no", "yes", "no", "yes", "no", "yes"]
        step = ActXGBoost()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertEqual(
            step.configuration["loss"]["categorical"],
            ["log_loss", "exponential"],
        )

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {"no", "yes"})
        self.assertTrue(set(predictions).issubset({"no", "yes"}))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

    def test_fit_multiclass_limits_loss_choices(self) -> None:
        df = self._make_features()
        y = [
            "class_a",
            "class_b",
            "class_c",
            "class_a",
            "class_b",
            "class_c",
            "class_a",
            "class_b",
            "class_c",
        ]
        step = ActXGBoost()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertIsNotNone(step.model)
        self.assertEqual(step.configuration["loss"]["categorical"], ["log_loss"])
        self.assertSetEqual(set(step.classes_), {"class_a", "class_b", "class_c"})

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActXGBoost()

        binary = self.make_dataset(df, y=["yes", "no", "yes", "no"])
        multiclass = self.make_dataset(df, y=["a", "b", "c", "a"])
        multilabel = self.make_dataset(
            df, y=[[True, False], [False, True], [True, True], [False, False]]
        )
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertTrue(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))
