"""Initial preprocessing belongs to every pipeline and is learned inside folds."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from iaml.actionables.predictors.classifier.act_decision_tree_classifier import ActDecisionTreeClassifier
from iaml.cache import Cache
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.logger import Logger
from iaml.meta_singleton import MetaSingleton
from iaml.metrics.accuracy_metric import AccuracyMetric
from iaml.sklearn_preprocessor import SklearnPreprocessor
from iaml.splitters import kfold_splitter


class RecordingEncoder(TransformerMixin, BaseEstimator):
    """Use a real OHE with fit/transform observations visible across deep copies."""

    fits = []
    unseen = []

    def __init__(self, prefix="encoded", invalid_output=None):
        self.prefix = prefix
        self.invalid_output = invalid_output

    def fit(self, X, y=None):
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        self.encoder_.fit(X[["category"]])
        self.categories_ = frozenset(self.encoder_.categories_[0])
        type(self).fits.append({"indices": tuple(X.index), "categories": self.categories_,
                               "columns": tuple(X.columns)})
        return self

    def transform(self, X):
        type(self).unseen.append(frozenset(X.category) - self.categories_)
        values = self.encoder_.transform(X[["category"]])
        result = pd.DataFrame(values, index=X.index,
                              columns=[f"{self.prefix}_{i}" for i in range(values.shape[1])])
        result["numeric"] = X.numeric.to_numpy(dtype=np.float32)
        if self.invalid_output == "array":
            return result.to_numpy()
        if self.invalid_output == "reorder":
            return result.iloc[::-1]
        if self.invalid_output == "text":
            result["text"] = "not encoded"
        if self.invalid_output == "bool":
            result["boolean"] = X.numeric > 3
        return result


def patient_two_folds(dataset):
    yield from kfold_splitter(dataset, nb_folds=2)


class InitialPreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, MetaSingleton, "_instances", dict(MetaSingleton._instances))
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        cache.configure(None)
        logger = Logger()
        self.addCleanup(setattr, logger, "verbose", logger.verbose)
        logger.verbose = 0
        RecordingEncoder.fits = []
        RecordingEncoder.unseen = []
        self.X = pd.DataFrame({"category": np.repeat([f"patient_{i}" for i in range(12)], 2),
                               "numeric": np.arange(24, dtype=float)}, index=np.arange(24) * 7 + 3)
        self.groups = pd.DataFrame({"patient": np.repeat(np.arange(12), 2)}, index=self.X.index)
        self.y = np.tile([False, True], 12)

    def fit_engine(self, template, **options):
        metric = AccuracyMetric()
        with patch.object(IAML, "default_pipeline"):
            engine = IAML(max_workers=1, max_duration=60, max_stage_duration=60,
                          splitter=patient_two_folds, main_metric=metric,
                          initial_preprocessor=template, **options)
        # Retain BOTH native generation branches, but use tiny trees instead of
        # an expensive AutoML search. Real grouped CV, pipeline fitting and final
        # selection remain in production code; only scheduling/search are mocked.
        engine.first_step = ActDecisionTreeClassifier()
        engine.first_step.configure({"max_depth": 1, "random_state": 42})
        engine.minimal_predictor_step = ActDecisionTreeClassifier()
        engine.minimal_predictor_step.configure({"max_depth": 2, "random_state": 42})
        for step in (engine.first_step, engine.minimal_predictor_step):
            step.use_cache = False

        def evaluate(candidates, dataset, **kwargs):
            for candidate in candidates:
                self.assertIsInstance(candidate.pipeline.transformers[0][1], SklearnPreprocessor)
                self.assertTrue(candidate.training_evaluate(dataset, splitter=patient_two_folds,
                                                             cache_split=False))
            return candidates

        with (patch("iaml.iaml.TimedPoolExecutor"),
              patch.object(engine, "_IAML__metrics_selection", return_value=[metric]),
              patch.object(engine, "_IAML__run_evaluations", side_effect=evaluate),
              patch.object(engine, "_IAML__optimize", side_effect=lambda ds, cands, **kw: cands)):
            candidates = engine.fit(self.X, self.y, groups=self.groups,
                                    generation_sample_size=len(self.X), n_candidates=2, verbose=0)
        return engine, candidates

    def test_both_generation_branches_keep_mandatory_preprocessing(self):
        template = RecordingEncoder()
        before = self.X.copy(deep=True)
        engine, candidates = self.fit_engine(template)
        self.assertEqual(len(candidates), 2)
        self.assertEqual({candidate.pipeline.predictor[1].get_config("max_depth") for candidate in candidates}, {1, 2})
        for candidate in candidates:
            step = candidate.pipeline.transformers[0][1]
            self.assertIsInstance(step, SklearnPreprocessor)
            self.assertFalse(step.can_be_disabled)
            self.assertFalse(step.is_interchangeable)
            self.assertNotIn(step, candidate.pipeline.optimizable_step)
            step.enable = False
            self.assertTrue(step.enable)
            self.assertEqual(step.transformer_.categories_, frozenset(self.X.category))
            self.assertEqual(candidate.predict_proba(self.X).shape, (24, 2))
        self.assertFalse(hasattr(template, "encoder_"))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in engine.init_candidate.dataset.X.dtypes))
        pd.testing.assert_frame_equal(self.X, before)

    def test_real_grouped_cv_fits_only_fold_categories_and_keeps_groups_out_of_features(self):
        _, candidates = self.fit_engine(RecordingEncoder())
        raw = Dataset(self.X, self.y, groups=self.groups)
        expected = {tuple(train.X.index): frozenset(train.X.category)
                    for train, test in patient_two_folds(raw)}
        records = [record for record in RecordingEncoder.fits if len(record["indices"]) < len(self.X)]
        self.assertGreaterEqual(len(records), 4)  # Two real folds in both branches.
        for record in records:
            self.assertIn(record["indices"], expected)
            self.assertEqual(record["categories"], expected[record["indices"]])
            self.assertEqual(set(record["columns"]), {"category", "numeric"})
        self.assertTrue(any(RecordingEncoder.unseen), "Fold validation must exercise unseen patient categories")
        self.assertTrue(all(candidate.computed_metrics for candidate in candidates))

    def test_sampled_refit_encoder_uses_the_same_sample_as_final_predictor(self):
        _, candidates = self.fit_engine(RecordingEncoder(), train_on_n_samples=12, refit_on_sample=True)
        sample = Dataset(self.X, self.y, groups=self.groups).sample(12)
        for candidate in candidates:
            encoder = candidate.pipeline.transformers[0][1].transformer_
            self.assertEqual(encoder.categories_, frozenset(sample.X.category))
            self.assertEqual(candidate.pipeline.predictor[1].model.tree_.n_node_samples[0], 12)

    def test_serialized_pipeline_retains_encoder_and_accepts_unknown_categories(self):
        _, candidates = self.fit_engine(RecordingEncoder())
        original = candidates[0]
        restored = cloudpickle.loads(cloudpickle.dumps(original))
        future = self.X.iloc[:3].copy()
        future["category"] = ["unknown A", "unknown B", "patient_0"]
        expected = original.predict_proba(future)
        np.testing.assert_allclose(restored.predict_proba(future), expected)
        self.assertTrue(np.isfinite(expected).all())

    def test_refit_resets_categories_and_cache_fingerprint_covers_template_parameters(self):
        template = RecordingEncoder().fit(self.X)
        step = SklearnPreprocessor(template)
        original_fingerprint = step.fingerprint()
        self.assertEqual(original_fingerprint, deepcopy(step).fingerprint())
        self.assertNotEqual(original_fingerprint, SklearnPreprocessor(RecordingEncoder(prefix="other")).fingerprint())
        self.assertFalse(hasattr(step.transformer, "encoder_"))
        step.fit(Dataset(self.X.iloc[:8], self.y[:8]))
        first = step.transformer_
        step.fit(Dataset(self.X.iloc[8:16], self.y[8:16]))
        self.assertIsNot(first, step.transformer_)
        self.assertFalse(first.categories_ & step.transformer_.categories_)
        self.assertEqual(step.fingerprint(), original_fingerprint)
        self.assertEqual(template.categories_, frozenset(self.X.category))

    def test_invalid_transformer_outputs_cannot_silently_change_row_alignment_or_types(self):
        dataset = Dataset(self.X, self.y)
        for output, exception in (("array", TypeError), ("reorder", ValueError), ("text", TypeError)):
            with self.subTest(output=output):
                step = SklearnPreprocessor(RecordingEncoder(invalid_output=output)).fit(dataset)
                with self.assertRaises(exception):
                    step.transform(self.X)
        result = SklearnPreprocessor(RecordingEncoder(invalid_output="bool")).fit(dataset).transform(self.X)
        self.assertEqual(result.boolean.dtype, np.dtype(np.float32))


if __name__ == "__main__":
    unittest.main()
