"""The final fitted model must use the requested training population."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.actionables.predictors.survival.act_survival_tree import ActSurvivalTree
from iaml.cache import Cache
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.logger import Logger
from iaml.meta_singleton import MetaSingleton
from iaml.metrics.accuracy_metric import AccuracyMetric
from iaml.metrics.concordance_index_metric import ConcordanceIndexMetric
from iaml.splitters import kfold_splitter


def positional_two_folds(dataset):
    """Use real CV; these tests exercise refit selection, not grouped splitting."""
    yield from kfold_splitter(Dataset(dataset.X, dataset.y), nb_folds=2)


class TestRefitSampling(unittest.TestCase):
    def setUp(self):
        # IAML.fit resets the singleton registry before the final fit.
        self.addCleanup(setattr, MetaSingleton, "_instances", dict(MetaSingleton._instances))
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        cache.configure(None)
        logger = Logger()
        self.addCleanup(setattr, logger, "verbose", logger.verbose)
        logger.verbose = 0
        self.X = pd.DataFrame({"row_number": np.arange(24, dtype=float)})
        # Boolean labels stay categorical even in the six-row CV training folds.
        self.y = np.tile([False, True], 12)

    def fit_small_model(self, X=None, y=None, *, groups_columns=None,
                        survival=False, force_downsize=False, **options):
        """Keep production generation/refit and real CV, with no process pool."""
        X = self.X if X is None else X
        y = self.y if y is None else y
        step_class = ActSurvivalTree if survival else ActDecisionTreeClassifier
        metric = ConcordanceIndexMetric() if survival else AccuracyMetric()
        with patch.object(IAML, "default_pipeline"):
            engine = IAML(
                max_workers=1, max_duration=60, max_stage_duration=60,
                splitter=positional_two_folds, main_metric=metric, **options,
            )
        engine.first_step = step_class()
        engine.first_step.configure({"max_depth": 3, "random_state": 42})
        engine.first_step.use_cache = False
        engine.minimal_predictor_step = None
        training_inputs = []
        evaluation_datasets = []
        native_fit = step_class.fit

        def record_real_fit(step, dataset):
            training_inputs.append((dataset.X.copy(), dataset.y.copy()))
            return native_fit(step, dataset)

        def evaluate(candidates, dataset, **kwargs):
            evaluation_datasets.append(dataset)
            if force_downsize and len(evaluation_datasets) == 1:
                return []
            for candidate in candidates:
                scores = candidate.training_evaluate(
                    dataset, splitter=engine.splitter, cache_split=False,
                )
                self.assertTrue(scores, "The real two-fold evaluation must succeed")
            return candidates

        with (
            patch("iaml.iaml.TimedPoolExecutor"),
            patch.object(engine, "_IAML__metrics_selection", return_value=[metric]),
            patch.object(engine, "_IAML__run_evaluations", side_effect=evaluate),
            patch.object(engine, "_IAML__optimize", side_effect=lambda ds, cands, **kw: cands),
            patch.object(step_class, "fit", autospec=True, side_effect=record_real_fit),
        ):
            candidates = engine.fit(
                X, y, groups_columns=groups_columns, generation_sample_size=12, verbose=0,
            )

        self.assertEqual(len(candidates), 1)
        self.assertIs(candidates[0], engine.chosen_candidate)
        self.assertTrue(candidates[0].computed_metrics)
        fitted_X, fitted_y = training_inputs[-1]
        native_tree = engine.chosen_model.predictor[1].model.tree_
        self.assertEqual(native_tree.n_node_samples[0], len(fitted_X))
        return engine, fitted_X, fitted_y, evaluation_datasets

    def assert_fitted_dataset(self, fitted_X, fitted_y, expected):
        pd.testing.assert_frame_equal(fitted_X, expected.X)
        np.testing.assert_array_equal(fitted_y, expected.y)

    def test_default_refits_on_the_initial_sample(self):
        engine, fitted_X, fitted_y, evaluations = self.fit_small_model(train_on_n_samples=12)

        self.assertTrue(engine.refit_on_sample)
        self.assertEqual(len(fitted_X), 12)
        self.assert_fitted_dataset(fitted_X, fitted_y, evaluations[0])
        # Refit must use the initial selection, not sample its already sampled rows again.
        expected = Dataset(self.X, self.y).sample(12)
        self.assert_fitted_dataset(fitted_X, fitted_y, expected)
        self.assertEqual(engine.chosen_model.predict(self.X).shape, (24,))

    def test_explicit_sample_preserves_position_with_duplicate_indices_and_groups(self):
        X = self.X.copy()
        X.index = np.repeat(np.arange(8), 3)
        X["subject"] = np.repeat(np.arange(12), 2)
        y = pd.Series(self.y, index=X.index)
        original_X, original_y = X.copy(), y.copy()

        _, fitted_X, fitted_y, evaluations = self.fit_small_model(
            X, y, groups_columns=["subject"], train_on_n_samples=12, refit_on_sample=True,
        )

        self.assertEqual(len(fitted_X), 12)
        self.assertNotIn("subject", fitted_X)
        self.assert_fitted_dataset(fitted_X, fitted_y, evaluations[0])
        positions = fitted_X["row_number"].to_numpy(dtype=int)
        np.testing.assert_array_equal(fitted_y, self.y[positions])
        pd.testing.assert_frame_equal(X, original_X)
        pd.testing.assert_series_equal(y, original_y)

    def test_false_refits_on_all_rows_and_excludes_group_columns(self):
        X = self.X.assign(subject=np.repeat(np.arange(12), 2))
        engine, fitted_X, fitted_y, evaluations = self.fit_small_model(
            X, groups_columns=["subject"], train_on_n_samples=12, refit_on_sample=False,
        )

        self.assertEqual(len(evaluations[0].X), 12)
        self.assertEqual(len(fitted_X), 24)
        pd.testing.assert_frame_equal(fitted_X, self.X)
        np.testing.assert_array_equal(fitted_y, self.y)
        self.assertEqual(engine.chosen_model._trained_columns, ["row_number"])
        self.assertIn("subject", X)

    def test_missing_or_nonpositive_cap_refits_on_all_rows_for_either_flag(self):
        for cap in (None, 0, -1):
            for flag in (False, True):
                with self.subTest(cap=cap, refit_on_sample=flag):
                    _, fitted_X, fitted_y, _ = self.fit_small_model(
                        train_on_n_samples=cap, refit_on_sample=flag,
                    )
                    self.assertEqual(len(fitted_X), 24)
                    pd.testing.assert_frame_equal(fitted_X, self.X)
                    np.testing.assert_array_equal(fitted_y, self.y)

    def test_cap_at_or_above_dataset_size_preserves_all_rows_without_duplication(self):
        for cap in (24, 30):
            with self.subTest(cap=cap):
                X = self.X.assign(subject=np.repeat(np.arange(12), 2))
                _, fitted_X, fitted_y, _ = self.fit_small_model(
                    X, groups_columns=["subject"], train_on_n_samples=cap,
                    refit_on_sample=True,
                )
                self.assertEqual(len(fitted_X), 24)
                pd.testing.assert_frame_equal(fitted_X, self.X)
                np.testing.assert_array_equal(fitted_y, self.y)

    def test_survival_refit_preserves_events_and_times_with_duplicate_indices(self):
        X = self.X.copy()
        X.index = np.repeat(np.arange(8), 3)
        y = [(row % 3 != 0, row + 0.25) for row in range(24)]

        engine, fitted_X, fitted_y, evaluations = self.fit_small_model(
            X, y, survival=True, train_on_n_samples=12, refit_on_sample=True,
        )

        self.assertEqual(len(fitted_X), 12)
        self.assert_fitted_dataset(fitted_X, fitted_y, evaluations[0])
        positions = fitted_X["row_number"].to_numpy(dtype=int)
        np.testing.assert_array_equal(fitted_y, np.asarray(y)[positions])
        model = engine.chosen_model.predictor[1].model
        np.testing.assert_array_equal(model.unique_times_, np.sort(fitted_y[:, 1]))
        self.assertTrue(np.isfinite(engine.chosen_model.predict(X)).all())

    def test_automatic_search_downsizing_keeps_the_original_refit_sample(self):
        # 500 rows is the production threshold for automatic downsizing. One
        # feature and a depth-three tree keep this branch cheap to exercise.
        X = pd.DataFrame({"row_number": np.arange(520, dtype=float)})
        y = np.tile([0, 1], 260)
        _, fitted_X, fitted_y, evaluations = self.fit_small_model(
            X, y, train_on_n_samples=500, refit_on_sample=True, force_downsize=True,
        )

        self.assertEqual([len(dataset.X) for dataset in evaluations], [500, 50])
        self.assertEqual(len(fitted_X), 500)
        self.assert_fitted_dataset(fitted_X, fitted_y, evaluations[0])
        np.testing.assert_array_equal(fitted_y, y[fitted_X["row_number"].to_numpy(dtype=int)])
