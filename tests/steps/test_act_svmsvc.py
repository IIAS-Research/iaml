"""Tests for ActSVMSVC step."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_svm_svc import ActSVMSVC


class TestActSVMSVC(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                "f3": [2.5, 1.5, 3.0, 0.5, 4.0, 2.0],
            }
        )

    def test_fit_predict_and_proba(self) -> None:
        df = self._make_features()
        y = [0, 1, 0, 1, 1, 0]
        step = ActSVMSVC()
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
        y = [0, 1, 0, 1, 1, 0]
        step = ActSVMSVC()
        step.configure("kernel", "linear")
        step.configure("random_state", 7)
        step.configure("tol", 1e-4)
        step.configure("class_weight", "balanced")
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params["kernel"], "linear")
        self.assertEqual(params["random_state"], 7)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertAlmostEqual(params["tol"], 1e-4)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActSVMSVC()

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
