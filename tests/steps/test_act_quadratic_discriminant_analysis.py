"""Tests for ActQuadraticDiscriminantAnalysis."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_quadratic_discriminant_analysis import (
    ActQuadraticDiscriminantAnalysis,
)


class TestActQuadraticDiscriminantAnalysis(StepTestCase):
    @staticmethod
    def _classification_data() -> tuple[pd.DataFrame, list[str]]:
        X = pd.DataFrame(
            {
                "f1": [0.0, 0.2, 0.1, 1.0, 1.2, 1.1],
                "f2": [1.0, 1.1, 0.9, 2.0, 2.1, 1.9],
            }
        )
        y = ["a", "a", "a", "b", "b", "b"]
        return X, y

    def test_fit_predicts_classes(self) -> None:
        X, y = self._classification_data()
        dataset = self.make_dataset(X, y)
        step = ActQuadraticDiscriminantAnalysis()

        self.fit_step(step, dataset)

        self.assertIsNotNone(step.model)
        preds = step.predict(X)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(set(preds).issubset(set(y)))
        self.assertEqual(set(step.classes_), set(y))

    def test_configure_reg_param_passthrough(self) -> None:
        X, y = self._classification_data()
        dataset = self.make_dataset(X, y)
        step = ActQuadraticDiscriminantAnalysis()
        step.configure("reg_param", 0.2)

        self.fit_step(step, dataset)

        self.assertAlmostEqual(step.model.reg_param, 0.2)

    def test_suitable_accepts_binary_target(self) -> None:
        X, y = self._classification_data()
        dataset = self.make_dataset(X, y)
        step = ActQuadraticDiscriminantAnalysis()

        self.assertTrue(step.suitable(dataset))

    def test_suitable_rejects_continuous_target(self) -> None:
        X = pd.DataFrame(
            {
                "f1": [0.0, 0.1, 0.2, 0.3, 0.4],
                "f2": [1.0, 1.1, 1.2, 1.3, 1.4],
            }
        )
        y = [0.1, 0.2, 0.3, 0.4, 0.5]
        dataset = self.make_dataset(X, y)
        step = ActQuadraticDiscriminantAnalysis()

        self.assertFalse(step.suitable(dataset))
