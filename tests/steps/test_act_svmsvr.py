"""Tests for ActSVMSVR step."""
import numpy as np
import pandas as pd
from sklearn import svm

from .step_test_case import StepTestCase

from iaml.actionables.predictors.regressor.act_svm_svr import ActSVMSVR


class TestActSVMSVR(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0],
                "f2": [1.0, 0.5, 1.5, 0.0, 2.0],
                "f3": [0.2, 0.4, 0.1, 0.6, 0.3],
            }
        )

    def test_fit_sets_model_and_predicts(self) -> None:
        df = self._make_features()
        y = [0.5, 0.7, 1.0, 1.3, 1.6]
        step = ActSVMSVR()
        dataset = self.make_dataset(df, y)

        self.assertIsNone(step.predict(df))

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, svm.SVR)

        predictions = step.predict(df)
        predictions_array = np.asarray(predictions)
        self.assertEqual(predictions_array.shape, (len(df),))
        self.assertTrue(np.isfinite(predictions_array).all())

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [1.0, 1.1, 1.3, 1.6, 2.0]
        step = ActSVMSVR()
        step.configure("kernel", "linear")
        step.configure("epsilon", 0.05)
        step.configure("tol", 1e-4)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params["kernel"], "linear")
        self.assertAlmostEqual(params["epsilon"], 0.05)
        self.assertAlmostEqual(params["tol"], 1e-4)

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActSVMSVR()

        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])
        categorical = self.make_dataset(df, y=["low", "high", "low", "high"])

        self.assertTrue(step.suitable(continuous))
        self.assertFalse(step.suitable(categorical))
