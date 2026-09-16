"""Regression tests for the complete inputs used by the pipeline cache."""
from threading import RLock
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from iaml.actionables.cleaning.act_target_encoder import ActTargetEncoder
from iaml.cache import Cache
from iaml.data_type import DataType
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.shared_cache import CacheService
from iaml.step import Step


class _TargetMeanShift(Step):
    """A supervised transform with externally observable execution counts."""

    fit_calls = 0
    transform_calls = 0

    def __init__(self):
        super().__init__()
        self.mean = None

    def fit(self, dataset):
        type(self).fit_calls += 1
        self.mean = float(np.mean(dataset.y))
        return self

    def transform(self, X):
        type(self).transform_calls += 1
        if self.mean is None:
            raise RuntimeError("The cached fitted step was not restored")
        result = X.copy()
        result["x"] = result["x"] + self.mean
        return result


class _PositiveRows(Step):
    """A resampler whose selected observations depend on y."""

    resample_calls = 0

    def resample(self, X, y):
        type(self).resample_calls += 1
        selected = np.flatnonzero(y > 0)
        return X.iloc[selected].copy(), y[selected].copy()


class _GroupMarker(Step):
    """Expose the groups received after an earlier preprocessing step."""

    def __init__(self):
        super().__init__()
        self.groups_seen = None

    def fit(self, dataset):
        self.groups_seen = dataset.groups.copy() if dataset.groups is not None else None
        return self

    def transform(self, X):
        result = X.copy()
        result["group"] = self.groups_seen.iloc[:, 0].to_numpy()
        return result


class _InPlaceResampler(Step):
    """Exercise a resampler that modifies both arguments before returning them."""

    resample_calls = 0

    def resample(self, X, y):
        type(self).resample_calls += 1
        X.loc[:, "x"] += 10
        y[:] += 1
        return X, y


class _InPlaceFit(_TargetMeanShift):
    """Exercise storage of the fitted step under its original input key."""

    def fit(self, dataset):
        dataset.X.loc[:, "x"] += 10
        dataset.y[:] += 1
        return super().fit(dataset)


class TestDatasetFingerprint(unittest.TestCase):
    """All mutable inputs that affect preprocessing belong to the cache key."""

    def setUp(self):
        self.X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=[10, 20, 30, 40])

    def test_equal_datasets_have_equal_fingerprints(self):
        dataset = Dataset(self.X, [0, 0, 0, 1], groups=pd.DataFrame({"id": [1, 1, 2, 2]}))
        self.assertEqual(dataset.fingerprint(), dataset.copy().fingerprint())

    def test_targets_distinguish_otherwise_equal_datasets(self):
        first = Dataset(self.X, [0, 0, 0, 1])
        second = Dataset(self.X, [1, 1, 1, 0])
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_list_and_array_targets_are_supported(self):
        for values in ([0, 0, 0, 1], [(True, 1.0), (False, 2.0),
                                     (True, 3.0), (False, 4.0)]):
            with self.subTest(values=values):
                from_list = Dataset(self.X, values)
                from_array = Dataset(self.X, np.array(values))
                self.assertEqual(from_list.fingerprint(), from_array.fingerprint())
                changed = from_array.copy()
                changed.y[-1] = changed.y[0]
                self.assertNotEqual(from_list.fingerprint(), changed.fingerprint())

    def test_missing_target_is_supported(self):
        first = Dataset(self.X)
        self.assertEqual(first.fingerprint(), Dataset(self.X.copy()).fingerprint())
        self.assertNotEqual(first.fingerprint(), Dataset(self.X, [0, 0, 0, 1]).fingerprint())

    def test_mutations_after_hashing_invalidate_every_dataset_input(self):
        def change_X(dataset):
            dataset.X.iloc[0, 0] = 99

        def change_y(dataset):
            dataset.y[0] = 1

        def change_groups(dataset):
            dataset.groups.iloc[0, 0] = 99

        def change_column_types(dataset):
            dataset.columns_types["x"] = (dataset.X["x"].dtype, DataType.CATEGORICAL)

        def change_target_type(dataset):
            dataset.type_of_target = "binary"

        for change in (change_X, change_y, change_groups, change_column_types, change_target_type):
            with self.subTest(change=change.__name__):
                dataset = Dataset(self.X.copy(), [0, 0, 0, 1],
                                  groups=pd.DataFrame({"id": [1, 1, 2, 2]}))
                before = dataset.fingerprint()
                change(dataset)
                self.assertNotEqual(before, dataset.fingerprint())


class TestPipelineCache(unittest.TestCase):
    """Use the real cache service with an in-process backend and no manager."""

    def setUp(self):
        cache = Cache()
        previous_backend = cache._backend
        self.addCleanup(cache.configure, previous_backend)
        self.saved = {}
        self.lru = []
        backend = CacheService()
        backend.__set_backend__(self.saved, self.lru, SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)
        for step_class in (_TargetMeanShift, _InPlaceFit):
            step_class.fit_calls = 0
            step_class.transform_calls = 0
        _PositiveRows.resample_calls = 0
        _InPlaceResampler.resample_calls = 0
        self.X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        self.y = np.array([0, 0, 0, 1])

    @staticmethod
    def pipeline(*steps):
        return IAMLPipeline([(str(i), step) for i, step in enumerate(steps)],
                            estimator_type="classifier")

    def evict_apply_results(self):
        apply_keys = [key for key in self.saved if key[0].startswith("apply_")]
        self.assertTrue(apply_keys)
        for key in apply_keys:
            del self.saved[key]
            self.lru.remove(key)

    def test_dataset_and_precomputed_cache_keys_are_interchangeable(self):
        dataset = Dataset(self.X, self.y)
        Cache().add_to_cache("example", dataset, {"value": [1]})
        self.assertEqual(Cache().from_cache("example", dataset.copy()), {"value": [1]})
        self.assertEqual(Cache().from_cache("example", dataset.fingerprint()), {"value": [1]})
        self.assertIsNone(Cache().from_cache("example", Dataset(self.X, 1 - self.y)))

    def test_dataframe_cache_arguments_remain_supported(self):
        Cache().add_to_cache("legacy", self.X, {"value": [1]})
        self.assertEqual(Cache().from_cache("legacy", self.X.copy()), {"value": [1]})
        changed = self.X.copy()
        changed.iloc[0, 0] += 1
        self.assertIsNone(Cache().from_cache("legacy", changed))

    def test_changed_targets_refit_supervised_transform_and_preserve_current_y(self):
        first = self.pipeline(_TargetMeanShift())
        first_X, first_y = first.fit_transform(self.X.copy(), self.y.copy())
        second = self.pipeline(_TargetMeanShift())
        second_X, second_y = second.fit_transform(self.X.copy(), 1 - self.y)

        pd.testing.assert_frame_equal(first_X, self.X + 0.25)
        pd.testing.assert_frame_equal(second_X, self.X + 0.75)
        np.testing.assert_array_equal(first_y, self.y)
        np.testing.assert_array_equal(second_y, 1 - self.y)
        self.assertEqual(second.transformers[0][1].mean, 0.75)
        self.assertEqual(_TargetMeanShift.fit_calls, 2)
        self.assertEqual(_TargetMeanShift.transform_calls, 2)

    def test_identical_inputs_reuse_cache_and_return_independent_copies(self):
        first = self.pipeline(_TargetMeanShift())
        first_X, first_y = first.fit_transform(self.X.copy(), self.y.copy())
        first_X.iloc[0, 0] = 99
        first_y[0] = 99
        first.transformers[0][1].mean = 99

        second = self.pipeline(_TargetMeanShift())
        second_X, second_y = second.fit_transform(self.X.copy(), self.y.copy())
        pd.testing.assert_frame_equal(second_X, self.X + 0.25)
        np.testing.assert_array_equal(second_y, self.y)
        self.assertEqual(second.transformers[0][1].mean, 0.25)
        self.assertIsNot(first.transformers[0][1], second.transformers[0][1])

        second_X.iloc[0, 0] = 88
        second_y[0] = 88
        third = self.pipeline(_TargetMeanShift())
        third_X, third_y = third.fit_transform(self.X.copy(), self.y.copy())
        pd.testing.assert_frame_equal(third_X, self.X + 0.25)
        np.testing.assert_array_equal(third_y, self.y)
        self.assertEqual(_TargetMeanShift.fit_calls, 1)
        self.assertEqual(_TargetMeanShift.transform_calls, 1)

    def test_fit_cache_hit_restores_step_when_apply_result_was_evicted(self):
        first = self.pipeline(_TargetMeanShift())
        expected_X, expected_y = first.fit_transform(self.X.copy(), self.y.copy())
        self.evict_apply_results()

        second = self.pipeline(_TargetMeanShift())
        actual_X, actual_y = second.fit_transform(self.X.copy(), self.y.copy())
        pd.testing.assert_frame_equal(actual_X, expected_X)
        np.testing.assert_array_equal(actual_y, expected_y)
        self.assertEqual(_TargetMeanShift.fit_calls, 1)
        self.assertEqual(_TargetMeanShift.transform_calls, 2)

    def test_restored_target_encoder_preserves_out_of_fold_training_encoding(self):
        X = pd.DataFrame({"category": ["a", "b", "c", "d"]})
        first_encoder = ActTargetEncoder()
        first_encoder.configure({"n_splits": 2})
        expected_X, expected_y = self.pipeline(first_encoder).fit_transform(X.copy(), self.y.copy())
        np.testing.assert_allclose(expected_X["category"], [0.25] * 4)
        self.evict_apply_results()

        second_encoder = ActTargetEncoder()
        second_encoder.configure({"n_splits": 2})
        second = self.pipeline(second_encoder)
        actual_X, actual_y = second.fit_transform(X.copy(), self.y.copy())
        pd.testing.assert_frame_equal(actual_X, expected_X)
        np.testing.assert_array_equal(actual_y, expected_y)
        self.assertIsNot(second.transformers[0][1], second_encoder)

    def test_resampler_does_not_reuse_rows_selected_with_other_targets(self):
        first_X, first_y = self.pipeline(_PositiveRows()).fit_transform(
            self.X.copy(), self.y.copy())
        second_X, second_y = self.pipeline(_PositiveRows()).fit_transform(
            self.X.copy(), 1 - self.y)

        pd.testing.assert_frame_equal(first_X, self.X.iloc[[3]])
        pd.testing.assert_frame_equal(second_X, self.X.iloc[[0, 1, 2]])
        np.testing.assert_array_equal(first_y, [1])
        np.testing.assert_array_equal(second_y, [1, 1, 1])
        self.assertEqual(_PositiveRows.resample_calls, 2)

    def test_groups_survive_transforms_and_distinguish_cached_inputs(self):
        for groups in ([1, 1, 2, 2], [3, 4, 4, 3], [3, 4, 4, 3]):
            with self.subTest(groups=groups):
                X = self.X.assign(subject=groups)
                pipeline = self.pipeline(_TargetMeanShift(), _GroupMarker())
                result, _ = pipeline.fit_transform(X, self.y.copy(), groups_columns=["subject"])
                np.testing.assert_array_equal(result["group"], groups)
                np.testing.assert_array_equal(
                    pipeline.transformers[1][1].groups_seen["subject"], groups)
                self.assertNotIn("subject", result.columns)
        self.assertEqual(_TargetMeanShift.fit_calls, 2)
        self.assertEqual(_TargetMeanShift.transform_calls, 2)

    def test_groups_follow_resampled_rows_before_later_transform(self):
        pipeline = self.pipeline(_PositiveRows(), _GroupMarker())
        result, y = pipeline.fit_transform(self.X.assign(subject=[10, 20, 30, 40]),
                                           self.y.copy(), groups_columns=["subject"])
        np.testing.assert_array_equal(result["group"], [40])
        np.testing.assert_array_equal(y, [1])
        self.assertNotIn("subject", result.columns)

    def test_in_place_application_is_cached_under_unmodified_input(self):
        expected_X = self.X + 10
        expected_y = self.y + 1
        for _ in range(2):
            actual_X, actual_y = self.pipeline(_InPlaceResampler()).fit_transform(
                self.X.copy(), self.y.copy())
            pd.testing.assert_frame_equal(actual_X, expected_X)
            np.testing.assert_array_equal(actual_y, expected_y)
        self.assertEqual(_InPlaceResampler.resample_calls, 1)

    def test_in_place_fit_is_cached_under_unmodified_input(self):
        original = Dataset(self.X.copy(), self.y.copy())
        step = _InPlaceFit()
        fit_key = f"fit_{step.fingerprint()}"
        expected_X, expected_y = self.pipeline(step).fit_transform(self.X.copy(), self.y.copy())

        self.assertIsNotNone(Cache().from_cache(fit_key, original))
        self.evict_apply_results()
        second = self.pipeline(_InPlaceFit())
        actual_X, actual_y = second.fit_transform(self.X.copy(), self.y.copy())

        pd.testing.assert_frame_equal(actual_X, expected_X)
        np.testing.assert_array_equal(actual_y, expected_y)
        self.assertEqual(second.transformers[0][1].mean, 1.25)
        self.assertEqual(_InPlaceFit.fit_calls, 1)


if __name__ == "__main__":
    unittest.main()
