"""Fitted MICE copies stay lazy while retaining independent prediction state."""
from contextlib import ExitStack
from copy import deepcopy
import os
import pickle
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from iaml.actionables.cleaning.act_mice import ActMICEForestImputer
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.shared_cache import CacheService


class TestMICECopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # One small real MICE fit supplies every test. No process pool is used.
        environment = patch.dict(os.environ, {"IAML_MICE_JOBS": "1", "IAML_DIAG": ""})
        environment.start()
        cls.addClassCleanup(environment.stop)
        cls.addClassCleanup(setattr, Logger(), "verbose", Logger().verbose)
        Logger().verbose = 0
        cls.limits = threadpool_limits(limits=1)
        cls.addClassCleanup(cls.limits.restore_original_limits)

        values = np.arange(18, dtype=float)
        cls.X = pd.DataFrame({"a": values + 0.5, "b": (values * 3) % 11,
                              "c": values % 4})
        cls.X.loc[[2, 8, 14], "a"] = np.nan
        cls.X.loc[[1, 7, 13], "b"] = np.nan
        cls.y = np.tile([0, 1], 9)
        cls.probe = cls.X.iloc[[0, 2, 5, 8, 11, 14]].copy()
        cls.probe.loc[5, "b"] = np.nan

        source = ActMICEForestImputer()
        source.fit(Dataset(cls.X.copy(), cls.y.copy()))
        completed = source.transform(cls.X.copy())
        if source.kernel is None or completed.isna().any().any():
            raise AssertionError("The real MICE fixture did not produce a fitted imputer")
        model = ActDecisionTreeClassifier()
        model.configure("max_depth", 2)
        model.fit(Dataset(completed, cls.y.copy()))
        cls.model_bytes = pickle.dumps(model)

        # Capture the state after training preprocessing, before predictions.
        cls.step_bytes = pickle.dumps(source)
        cls.initial_rng = source.kernel._random_state.get_state()
        cls.expected_first = source.transform(cls.probe.copy())
        cls.first_rng = source.kernel._random_state.get_state()
        cls.expected_second = source.transform(cls.probe.copy())
        cls.second_rng = source.kernel._random_state.get_state()

    def setUp(self):
        self.step = pickle.loads(self.step_bytes)
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        backend = CacheService()
        backend.__set_backend__({}, [], SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)

    def assert_rng_equal(self, actual, expected):
        self.assertEqual(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[2:], expected[2:])

    def assert_frozen(self, step):
        self.assertIsNone(step._kernel)
        self.assertIsInstance(step._kernel_snapshot, bytes)

    def no_kernel_restoration(self, block_serialization=False):
        """Fail if a supposedly cheap operation reads Parquet or creates Boosters."""
        stack = ExitStack()
        forbidden = [
            "miceforest.imputation_kernel.ImputationKernel.__setstate__",
            "miceforest.imputation_kernel.read_parquet",
            "miceforest.imputed_data.read_parquet",
            "lightgbm.basic.Booster.model_from_string",
            "lightgbm.basic.Booster.__setstate__",
        ]
        if block_serialization:
            forbidden.extend([
                "pandas.DataFrame.to_parquet",
                "lightgbm.basic.Booster.model_to_string",
            ])
        for name in forbidden:
            stack.enter_context(patch(name, side_effect=AssertionError(f"Unexpected {name}")))
        return stack

    def test_candidate_copy_chains_share_snapshot_without_model_or_parquet_work(self):
        self.assert_frozen(self.step)
        snapshot = self.step._kernel_snapshot
        candidate = Candidate(
            Dataset(self.X.copy(), self.y.copy()),
            iaml_pipeline=IAMLPipeline([("mice", self.step)], estimator_type="classifier"),
        )
        self.step.candidate = candidate

        with self.no_kernel_restoration(block_serialization=True):
            for _ in range(12):
                candidate = candidate.to_input()
                copied = candidate.pipeline.transformers[0][1]
                self.assert_frozen(copied)
                self.assertIs(copied._kernel_snapshot, snapshot)
                self.assertIsNone(getattr(copied, "candidate", None))

        self.assertIs(self.step.candidate.pipeline.transformers[0][1], self.step)

    def test_first_transform_restores_expected_predictions_and_rng(self):
        self.assert_frozen(self.step)

        actual = self.step.transform(self.probe.copy())

        pd.testing.assert_frame_equal(actual, self.expected_first)
        self.assertIsNotNone(self.step._kernel)
        self.assertIsNone(self.step._kernel_snapshot)
        self.assert_rng_equal(self.step.kernel._random_state.get_state(), self.first_rng)
        kernel = self.step.kernel
        with self.no_kernel_restoration():
            self.assertIs(self.step.kernel, kernel)

    def test_clones_have_independent_models_and_random_state(self):
        first = deepcopy(self.step)
        second = deepcopy(self.step)
        pd.testing.assert_frame_equal(first.transform(self.probe.copy()), self.expected_first)
        pd.testing.assert_frame_equal(first.transform(self.probe.copy()), self.expected_second)
        self.assert_frozen(second)

        pd.testing.assert_frame_equal(second.transform(self.probe.copy()), self.expected_first)

        self.assertIsNot(first.kernel, second.kernel)
        self.assertIsNot(first.kernel._random_state, second.kernel._random_state)
        self.assert_rng_equal(first.kernel._random_state.get_state(), self.second_rng)
        self.assert_rng_equal(second.kernel._random_state.get_state(), self.first_rng)
        key = next(iter(first.kernel.models))
        self.assertIsNot(first.kernel.models[key], second.kernel.models[key])

    def test_copy_after_transform_captures_current_state_without_changing_source(self):
        self.step.transform(self.probe.copy())
        kernel = self.step.kernel

        copied = deepcopy(self.step)

        self.assert_frozen(copied)
        self.assertIs(self.step.kernel, kernel)
        self.assert_rng_equal(kernel._random_state.get_state(), self.first_rng)
        pd.testing.assert_frame_equal(self.step.transform(self.probe.copy()), self.expected_second)
        pd.testing.assert_frame_equal(copied.transform(self.probe.copy()), self.expected_second)
        self.assert_rng_equal(copied.kernel._random_state.get_state(), self.second_rng)
        self.assertIsNot(copied.kernel, kernel)

    def test_fit_discards_snapshot_before_skipping_or_constructing_a_new_kernel(self):
        skipped_inputs = (
            pd.DataFrame({"a": [1.0, np.nan, 3.0]}),
            pd.DataFrame({"a": [np.nan] * 6}),
            pd.DataFrame({"text": ["a", "b", "a", "b", "a", "b"]}),
        )
        for X in skipped_inputs:
            with self.subTest(columns=list(X.columns), rows=len(X)):
                step = pickle.loads(self.step_bytes)
                with self.no_kernel_restoration():
                    step.fit(Dataset(X, np.zeros(len(X))))
                    self.assertIsNone(step.kernel)
                self.assertIsNone(step._kernel_snapshot)

        # A valid refit must also discard the old state before constructing its
        # replacement. Stop at construction to avoid a second MICE training run.
        step = pickle.loads(self.step_bytes)

        class NewKernelRequested(Exception):
            pass

        def constructing_new_kernel(**kwargs):
            self.assertIsNone(step._kernel)
            self.assertIsNone(step._kernel_snapshot)
            pd.testing.assert_frame_equal(kwargs["data"], self.X.reset_index(drop=True))
            raise NewKernelRequested

        with self.no_kernel_restoration(), \
             patch("iaml.actionables.cleaning.act_mice.mf.ImputationKernel",
                   side_effect=constructing_new_kernel):
            with self.assertRaises(NewKernelRequested):
                step.fit(Dataset(self.X.copy(), self.y.copy()))

    def test_pickle_keeps_frozen_and_live_steps_ready_without_eager_restoration(self):
        with self.no_kernel_restoration(block_serialization=True):
            frozen = pickle.loads(pickle.dumps(self.step))
        self.assert_frozen(frozen)
        pd.testing.assert_frame_equal(frozen.transform(self.probe.copy()), self.expected_first)

        payload = pickle.dumps(frozen)
        with self.no_kernel_restoration():
            restored = pickle.loads(payload)
        self.assert_frozen(restored)
        pd.testing.assert_frame_equal(restored.transform(self.probe.copy()), self.expected_second)
        self.assert_rng_equal(restored.kernel._random_state.get_state(), self.second_rng)

    def test_cache_returns_lazy_independent_fitted_imputers(self):
        dataset = Dataset(self.X.copy(), self.y.copy())
        with self.no_kernel_restoration(block_serialization=True):
            Cache().add_to_cache("mice_copy", dataset, self.step)
            first = Cache().from_cache("mice_copy", dataset)
            second = Cache().from_cache("mice_copy", dataset)
        self.assert_frozen(first)
        self.assert_frozen(second)
        self.assertIs(first._kernel_snapshot, second._kernel_snapshot)
        pd.testing.assert_frame_equal(first.transform(self.probe.copy()), self.expected_first)
        first.transform(self.probe.copy())
        pd.testing.assert_frame_equal(second.transform(self.probe.copy()), self.expected_first)
        self.assertIsNot(first.kernel, second.kernel)

    def test_pipeline_copy_pickle_and_cache_predict_without_refitting(self):
        model = pickle.loads(self.model_bytes)
        pipeline = IAMLPipeline([("mice", self.step), ("tree", model)],
                                estimator_type="classifier")
        dataset = Dataset(self.X.copy(), self.y.copy())
        expected = model.predict(self.expected_first)
        with self.no_kernel_restoration(block_serialization=True):
            copied = pipeline.copy()
            restored = pickle.loads(pipeline.pickle())
            Cache().add_to_cache("mice_pipeline", dataset, pipeline)
            cached = Cache().from_cache("mice_pipeline", dataset)

        with patch.object(ActMICEForestImputer, "fit", side_effect=AssertionError("Unexpected MICE fit")), \
             patch.object(ActDecisionTreeClassifier, "fit", side_effect=AssertionError("Unexpected tree fit")):
            for label, candidate in (("copy", copied), ("pickle", restored), ("cache", cached)):
                with self.subTest(route=label):
                    self.assert_frozen(candidate.transformers[0][1])
                    np.testing.assert_array_equal(candidate.predict(self.probe.copy()), expected)
                    self.assert_rng_equal(candidate.transformers[0][1].kernel._random_state.get_state(),
                                          self.first_rng)


if __name__ == "__main__":
    unittest.main()
