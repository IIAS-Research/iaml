"""XGBoost adapters must accept ordinary pandas feature names safely."""
from functools import partial
import pickle
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from iaml.actionables.predictors.classifier.act_xgboost import ActXGBoost
from iaml.actionables.predictors.regressor.act_xgboost_regressor import ActXGBoostRegressor
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.metrics import AccuracyMetric, R2ScoreMetric
from iaml.splitters import kfold_splitter


class TestXGBoostFeatureNames(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame(
            np.random.default_rng(7).normal(size=(24, 5)),
            columns=["dose [mg]", "dose %5Bmg%5D", "age<65", "age%3C65", "normal"],
        )
        self.regression_target = 2 * self.X.iloc[:, 0].to_numpy() + self.X.iloc[:, 2].to_numpy()
        self.classification_target = np.where(self.regression_target > 0, "yes", "no")

    def models(self):
        for cls, target, metric, kind in (
            (ActXGBoost, self.classification_target, AccuracyMetric(), "classifier"),
            (ActXGBoostRegressor, self.regression_target, R2ScoreMetric(), "regressor"),
        ):
            model = cls()
            model.configure({"n_estimators": 5, "max_depth": 2})
            yield model, target, metric, kind

    def test_special_names_match_predictions_with_plain_names_without_mutating_data(self):
        original = self.X.copy(deep=True)
        plain = self.X.set_axis([f"feature_{index}" for index in range(self.X.shape[1])], axis=1)
        weights = np.arange(1, len(self.X) + 1)
        for model, target, _, kind in self.models():
            with self.subTest(kind=kind):
                dataset = Dataset(self.X, target)
                model.fit(dataset)
                baseline = type(model)()
                baseline.configure({"n_estimators": 5, "max_depth": 2})
                baseline.fit(Dataset(plain, target))
                predictions = model.predict(self.X)
                np.testing.assert_array_equal(predictions, baseline.predict(plain))
                scorer = accuracy_score if kind == "classifier" else r2_score
                self.assertAlmostEqual(model.score(self.X, target, sample_weight=weights),
                                       scorer(target, predictions, sample_weight=weights))
                if kind == "classifier":
                    np.testing.assert_allclose(model.predict_proba(self.X),
                                               baseline.predict_proba(plain))
                    expected_labels = model.classes_[model.predict_proba(self.X).argmax(axis=1)]
                    np.testing.assert_array_equal(predictions, expected_labels)
                pd.testing.assert_frame_equal(dataset.X, original)
                pd.testing.assert_frame_equal(self.X, original)

    def test_wrong_names_and_order_are_still_rejected(self):
        for model, target, _, kind in self.models():
            model.fit(Dataset(self.X, target))
            for frame in (self.X.iloc[:, ::-1], self.X.rename(columns={"normal": "wrong"})):
                with self.subTest(kind=kind, columns=list(frame.columns)):
                    with self.assertRaisesRegex(ValueError, "feature_names mismatch"):
                        model.predict(frame)
                    with self.assertRaisesRegex(ValueError, "feature_names mismatch"):
                        model.score(frame, target)
                    if kind == "classifier":
                        with self.assertRaisesRegex(ValueError, "feature_names mismatch"):
                            model.predict_proba(frame)

    def test_literal_escape_is_not_accepted_as_the_original_name(self):
        original = self.X[["dose [mg]"]]
        renamed = original.rename(columns={"dose [mg]": "dose %5Bmg%5D"})
        for model, target, _, kind in self.models():
            with self.subTest(kind=kind):
                model.fit(Dataset(original, target))
                with self.assertRaisesRegex(ValueError, "feature_names mismatch"):
                    model.predict(renamed)

    def test_numpy_prediction_remains_supported(self):
        for model, target, _, kind in self.models():
            with self.subTest(kind=kind):
                model.fit(Dataset(self.X, target))
                np.testing.assert_array_equal(model.predict(self.X.to_numpy()), model.predict(self.X))
                self.assertAlmostEqual(model.score(self.X.to_numpy(), target),
                                       model.score(self.X, target))
                if kind == "classifier":
                    np.testing.assert_allclose(model.predict_proba(self.X.to_numpy()),
                                               model.predict_proba(self.X))

    def test_integer_and_multiindex_columns_remain_supported(self):
        for columns in (pd.Index(range(self.X.shape[1])),
                        pd.MultiIndex.from_product([["medical"], self.X.columns])):
            frame = self.X.set_axis(columns, axis=1)
            original = frame.copy(deep=True)
            for model, target, _, kind in self.models():
                with self.subTest(kind=kind, columns=columns):
                    model.fit(Dataset(frame, target))
                    self.assertEqual(len(model.predict(frame)), len(frame))
                    self.assertTrue(np.isfinite(model.score(frame, target)))
                    pd.testing.assert_frame_equal(frame, original)

    def test_pipeline_evaluation_pickle_and_refit_preserve_public_names(self):
        for model, target, metric, kind in self.models():
            with self.subTest(kind=kind):
                dataset = Dataset(self.X, target)
                pipeline = IAMLPipeline([("xgboost", model)], estimator_type=kind)
                candidate = Candidate(dataset, metrics=[metric], main_metric=metric,
                                      iaml_pipeline=pipeline)
                scores = candidate.training_evaluate(
                    dataset, splitter=partial(kfold_splitter, nb_folds=3), cache_split=False,
                )
                self.assertTrue(np.isfinite(scores[str(metric)]))
                pipeline.fit(self.X, target)
                restored = pickle.loads(pipeline.pickle())
                np.testing.assert_array_equal(restored.predict(self.X), pipeline.predict(self.X))
                self.assertEqual(restored._trained_columns, list(self.X.columns))
                if kind == "classifier":
                    np.testing.assert_allclose(restored.predict_proba(self.X),
                                               pipeline.predict_proba(self.X))
                renamed = self.X.rename(columns={"dose [mg]": "new dose [mg]"})
                restored.fit(renamed, target)
                np.testing.assert_array_equal(restored.predict(renamed), pipeline.predict(self.X))
