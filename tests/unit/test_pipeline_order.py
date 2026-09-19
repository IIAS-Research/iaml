"""Regression tests for the order of transforms and training-only resamplers."""
import pickle
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.base import clone

from iaml.actionables.cleaning.act_simple_imputer import ActSimpleImputer
from iaml.actionables.imbalance.act_smote import ActSMOTE
from iaml.actionables.normalize.act_standard_scaler import ActStandardScaler
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.shared_cache import CacheService


class TestPipelineOrder(unittest.TestCase):
    """Exercise the real preprocessing steps with an isolated cache backend."""

    def setUp(self):
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        self.saved = {}
        self.lru = []
        backend = CacheService()
        backend.__set_backend__(self.saved, self.lru, SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)
        self.X = pd.DataFrame({"feature": np.arange(14, dtype=float) + 0.5})
        self.X.loc[0, "feature"] = np.nan
        self.y = np.array([0] * 10 + [1] * 4)

    def make_steps(self):
        steps = [ActSimpleImputer(), ActSMOTE(), ActStandardScaler(),
                 ActDecisionTreeClassifier()]
        for step in steps:
            self.addCleanup(step.reset_cache)
        return [(str(index), step) for index, step in enumerate(steps)]

    def make_pipeline(self):
        return IAMLPipeline(self.make_steps(), estimator_type="classifier")

    def generated_candidate(self):
        candidate = Candidate(Dataset(self.X.copy(), self.y.copy()))
        for _, step in self.make_steps():
            results = step.run(candidate)
            self.assertEqual(len(results), 1)
            candidate = results[0]
        return candidate

    def assert_training_order(self, pipeline):
        self.assertEqual(
            [type(step) for _, step in pipeline.training_steps],
            [ActSimpleImputer, ActSMOTE, ActStandardScaler, ActDecisionTreeClassifier],
        )
        self.assertEqual(
            [type(step) for _, step in pipeline.steps],
            [ActSimpleImputer, ActStandardScaler, ActDecisionTreeClassifier],
        )

    def test_generated_pipeline_refits_in_generation_order_and_only_resamples_training(self):
        candidate = self.generated_candidate()
        self.assertEqual(len(candidate.dataset.X), 20)
        self.assertFalse(candidate.dataset.X.isna().any().any())
        self.assert_training_order(candidate.pipeline)

        X, y = candidate.pipeline.fit_transform(self.X.copy(), self.y.copy())
        pd.testing.assert_frame_equal(X, candidate.dataset.X)
        np.testing.assert_array_equal(y, candidate.dataset.y)
        candidate.pipeline.fit(self.X.copy(), self.y.copy())
        self.assertEqual(
            [(step["class"], step["role"]) for step in candidate.pipeline_audit_summary()["steps"]],
            [("ActSimpleImputer", "transformer"), ("ActSMOTE", "resampler"),
             ("ActStandardScaler", "transformer"), ("ActDecisionTreeClassifier", "predictor")],
        )

        sampler = candidate.pipeline.resamplers[0][1]
        with patch.object(sampler, "resample", side_effect=AssertionError("Resampled inference")):
            predictions = candidate.predict(self.X.copy())
            probabilities = candidate.predict_proba(self.X.copy())
        self.assertEqual(predictions.shape, (len(self.X),))
        self.assertEqual(probabilities.shape, (len(self.X), 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(predictions, probabilities.argmax(axis=1))

    def test_constructor_and_steps_assignment_preserve_interleaved_order(self):
        candidate = self.generated_candidate()
        for use_setter in (False, True):
            with self.subTest(use_setter=use_setter):
                steps = self.make_steps()
                if use_setter:
                    pipeline = self.make_pipeline()
                    pipeline.steps = steps
                else:
                    pipeline = IAMLPipeline(steps, estimator_type="classifier")
                self.assert_training_order(pipeline)
                X, y = pipeline.fit_transform(self.X.copy(), self.y.copy())
                pd.testing.assert_frame_equal(X, candidate.dataset.X)
                np.testing.assert_array_equal(y, candidate.dataset.y)

    def test_copy_clone_and_pickle_preserve_order_and_fitted_predictions(self):
        pipeline = self.make_pipeline().fit(self.X.copy(), self.y.copy())
        expected = pipeline.predict(self.X.copy())
        for method, copied in (
            ("copy", pipeline.copy()),
            ("clone", clone(pipeline)),
            ("pickle", pickle.loads(pipeline.pickle())),
        ):
            with self.subTest(method=method):
                self.assertIsNot(copied, pipeline)
                self.assert_training_order(copied)
                for (_, original), (_, duplicate) in zip(
                    pipeline.training_steps, copied.training_steps
                ):
                    self.assertIsNot(duplicate, original)
                np.testing.assert_array_equal(copied.predict(self.X.copy()), expected)

    def test_replacing_and_removing_steps_preserves_the_remaining_positions(self):
        steps = self.make_steps()
        pipeline = IAMLPipeline(steps, estimator_type="classifier")
        replacement = ActSMOTE()
        self.addCleanup(replacement.reset_cache)
        self.assertTrue(pipeline.replace_step(steps[1][1], replacement))
        expected = [steps[0][1], replacement, steps[2][1], steps[3][1]]
        for (_, actual), original in zip(pipeline.training_steps, expected):
            self.assertIs(actual, original)

        self.assertTrue(pipeline.remove_step(steps[2][1]))
        self.assertEqual([id(step) for _, step in pipeline.training_steps],
                         [id(step) for step in (steps[0][1], replacement, steps[3][1])])
        self.assertTrue(pipeline.remove_step(replacement))
        self.assertTrue(pipeline.remove_step(steps[3][1]))
        self.assertEqual(pipeline.training_steps, [steps[0]])
        self.assertFalse(pipeline.remove_step(replacement))
        self.assertFalse(pipeline.replace_step(replacement, steps[1][1]))

    def test_fingerprints_include_interleaving_and_parameter_changes(self):
        steps = self.make_steps()
        forward = IAMLPipeline(steps, estimator_type="classifier")
        reverse = IAMLPipeline([steps[1], steps[0], *steps[2:]], estimator_type="classifier")
        for method in ("fingerprint", "transformers_resamplers_fingerprint"):
            with self.subTest(method=method):
                self.assertNotEqual(getattr(forward, method)(), getattr(reverse, method)())
                self.assertEqual(getattr(forward, method)(), getattr(forward.copy(), method)())

        before = (forward.fingerprint(), forward.transformers_resamplers_fingerprint())
        steps[1][1].configure({"k_neighbors": 2})
        self.assertNotEqual(forward.fingerprint(), before[0])
        self.assertNotEqual(forward.transformers_resamplers_fingerprint(), before[1])

    def test_cached_fitted_replacements_preserve_order_and_transformed_data(self):
        first = self.make_pipeline()
        expected_X, expected_y = first.fit_transform(self.X.copy(), self.y.copy())
        apply_keys = [key for key in self.saved if key[0].startswith("apply_")]
        self.assertTrue(apply_keys)
        for key in apply_keys:
            del self.saved[key]
            self.lru.remove(key)

        second = self.make_pipeline()
        original_steps = second.training_steps[:-1]
        X, y = second.fit_transform(self.X.copy(), self.y.copy())
        self.assert_training_order(second)
        pd.testing.assert_frame_equal(X, expected_X)
        np.testing.assert_array_equal(y, expected_y)
        for (_, original), (_, restored) in zip(original_steps, second.training_steps):
            self.assertIsNot(restored, original)


if __name__ == "__main__":
    unittest.main()
