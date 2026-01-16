"""Tests for VoidStep."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.step import Step
from iaml.void_step import VoidStep


class PredictOnlyStep(Step):
    name = "PredictOnlyStep"

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class TransformOnlyStep(Step):
    name = "TransformOnlyStep"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class ResampleOnlyStep(Step):
    name = "ResampleOnlyStep"

    def resample(self, X: pd.DataFrame, y: list) -> tuple[pd.DataFrame, list]:
        return X, y


class TestVoidStep(StepTestCase):
    def test_predict_passthrough_sets_flags_and_pipeline(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5], "c": ["x", "y"]})
        mimic = PredictOnlyStep()
        mimic.tags = {"predict"}
        step = VoidStep(step_to_mimic=mimic)

        result = step.predict(df)

        self.assertIs(result, df)
        self.assertEqual(step.tags, mimic.tags)
        self.assertTrue(step.is_interchangeable)
        self.assertTrue(step.optimizable)
        pipeline = step.json_pipeline()
        self.assertEqual(pipeline["step"], "VoidStep")
        self.assertEqual(pipeline["step_to_mimic"]["step"], "PredictOnlyStep")

    def test_transform_passthrough(self) -> None:
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        mimic = TransformOnlyStep()
        mimic.tags = {"transform"}
        step = VoidStep(step_to_mimic=mimic)

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)

    def test_resample_passthrough(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        y = [0, 1, 0]
        mimic = ResampleOnlyStep()
        mimic.tags = {"resample"}
        step = VoidStep(step_to_mimic=mimic)

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        self.assertFrameEqual(X_resampled, df)
        self.assertListEqual(list(y_resampled), y)

    def test_from_pipeline_requires_step_to_mimic(self) -> None:
        with self.assertRaises(TypeError):
            VoidStep.from_pipeline({"step": "VoidStep"})
