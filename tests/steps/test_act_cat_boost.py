"""Tests for ActCatBoost."""
from unittest.mock import patch

import pandas as pd
from catboost import CatBoostError

from .step_test_case import StepTestCase
from iaml.actionables.predictors.classifier.act_catboost_classifier import ActCatBoost


class TestActCatBoost(StepTestCase):
    def _configure_fast(self, step: ActCatBoost, *, multiclass: bool = False) -> None:
        config = {
            "iterations": 100,
            "learning_rate": 0.1,
            "depth": 2,
            "l2_leaf_reg": 1,
            "border_count": 32,
            "leaf_estimation_iterations": 1,
        }
        if multiclass:
            config["loss_function"] = "MultiClass"
            config["eval_metric"] = "Accuracy"
        step.configure(config)

    def test_fit_binary_trains_and_sets_loss_function_choices(self) -> None:
        X = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5],
                "f2": [1, 0, 1, 0, 1, 0],
            }
        )
        y = ["no", "yes", "no", "yes", "no", "yes"]
        dataset = self.make_dataset(X, y)
        step = ActCatBoost()
        self._configure_fast(step)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertEqual(
            step.configuration["loss_function"]["categorical"],
            ["Logloss", "CrossEntropy"],
        )
        self.assertListEqual(step.label_encoder.classes_.tolist(), ["no", "yes"])

    def test_fit_multiclass_updates_configuration(self) -> None:
        X = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "f2": [1, 1, 0, 0, 1, 0, 1, 0, 1],
            }
        )
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
        dataset = self.make_dataset(X, y)
        step = ActCatBoost()
        self._configure_fast(step, multiclass=True)

        self.fit_step(step, dataset)

        self.assertIsNotNone(step.model)
        self.assertEqual(
            step.configuration["loss_function"]["categorical"],
            ["MultiClass", "MultiClassOneVsAll"],
        )
        self.assertEqual(
            step.configuration["eval_metric"]["categorical"],
            ["AUC", "Accuracy"],
        )

    def test_fit_wraps_catboost_error(self) -> None:
        X = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3],
                "f2": [1, 0, 1, 0],
            }
        )
        y = [0, 1, 0, 1]
        dataset = self.make_dataset(X, y)
        step = ActCatBoost()
        self._configure_fast(step)

        with patch(
            "iaml.actionables.predictors.classifier.act_catboost_classifier.CatBoostClassifier.fit",
            side_effect=CatBoostError("boom"),
        ):
            with self.assertRaises(ValueError) as ctx:
                self.fit_step(step, dataset)

        self.assertIn("CatBoostClassifier training failed", str(ctx.exception))

    def test_suitable_binary_true_continuous_false(self) -> None:
        X = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5],
                "f2": [1, 0, 1, 0, 1, 0],
            }
        )
        step = ActCatBoost()

        binary_dataset = self.make_dataset(X, ["no", "yes", "no", "yes", "no", "yes"])
        continuous_dataset = self.make_dataset(
            X,
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )

        self.assertTrue(step.suitable(binary_dataset))
        self.assertFalse(step.suitable(continuous_dataset))
