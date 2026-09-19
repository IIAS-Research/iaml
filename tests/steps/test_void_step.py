"""Tests for VoidStep."""
from copy import deepcopy
import pickle
from threading import RLock
from types import SimpleNamespace

import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_preprocessing.act_pca import ActPCA
from iaml.actionables.imbalance.act_smote import ActSMOTE
from iaml.cache import Cache
from iaml.iaml_pipeline import IAMLPipeline
from iaml.shared_cache import CacheService
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

    def test_fingerprints_distinguish_operations_and_survive_copy_and_pickle(self) -> None:
        fingerprints = set()
        for mimic_type in (PredictOnlyStep, TransformOnlyStep, ResampleOnlyStep):
            with self.subTest(mimic=mimic_type.__name__):
                mimic = mimic_type()
                mimic.tags = {"family"}
                step = VoidStep(step_to_mimic=mimic)
                fingerprint = step.fingerprint()
                fingerprints.add(fingerprint)

                self.assertEqual(deepcopy(step).fingerprint(), fingerprint)
                self.assertEqual(pickle.loads(pickle.dumps(step)).fingerprint(), fingerprint)

        self.assertEqual(len(fingerprints), 3)

    def test_fingerprint_distinguishes_tags_of_the_same_mimic(self) -> None:
        first_mimic = TransformOnlyStep()
        first_mimic.tags = {"normalize"}
        second_mimic = TransformOnlyStep()
        second_mimic.tags = {"features_preprocessing"}

        self.assertNotEqual(
            VoidStep(step_to_mimic=first_mimic).fingerprint(),
            VoidStep(step_to_mimic=second_mimic).fingerprint(),
        )

    def test_cached_fit_and_refit_keep_distinct_placeholder_operations(self) -> None:
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        saved = {}
        backend = CacheService()
        backend.__set_backend__(saved, [], SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)

        pipeline = IAMLPipeline([
            ("imbalance", VoidStep(step_to_mimic=ActSMOTE())),
            ("preprocessing", VoidStep(step_to_mimic=ActPCA())),
        ], estimator_type="classifier")
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        y = [0, 1, 0, 1]
        fingerprint = pipeline.fingerprint()

        for fitting_pass in range(2):
            with self.subTest(fitting_pass=fitting_pass):
                actual_X, actual_y = pipeline.fit_transform(X.copy(), y.copy())

                self.assertFrameEqual(actual_X, X)
                self.assertListEqual(list(actual_y), y)
                self.assertEqual(len(pipeline.resamplers), 1)
                self.assertEqual(len(pipeline.transformers), 1)
                self.assertEqual(pipeline.resamplers[0][1].tags, {"imbalance"})
                self.assertEqual(pipeline.transformers[0][1].tags, {"features_preprocessing"})
                self.assertEqual(pipeline.fingerprint(), fingerprint)

        self.assertEqual(sum(key[0].startswith("fit_") for key in saved), 2)
