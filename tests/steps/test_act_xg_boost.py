"""Tests for ActXGBoost step."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from .step_test_case import StepTestCase

from iaml.actionables.predictors.classifier.act_xgboost import ActXGBoost
from iaml.candidate import Candidate
from iaml.metrics import AccuracyMetric
from iaml.splitters import kfold_splitter
from tests.helpers.datasets import make_classification_data


class TestActXGBoost(StepTestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7, 8],
                "f2": [1, 0, 1, 0, 1, 0, 1, 0, 1],
                "f3": [0.2, 0.4, 0.1, 0.3, 0.5, 0.7, 0.6, 0.8, 0.9],
            }
        )

    def _configure_fast(self, step: ActXGBoost) -> None:
        step.configure({"n_estimators": 5, "max_depth": 2, "random_state": 7})

    def test_fit_binary_uses_xgboost_and_preserves_original_labels(self) -> None:
        df = self._make_features().iloc[:6].copy()
        y = ["no", "yes", "no", "yes", "no", "yes"]
        step = ActXGBoost()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsInstance(step.model, XGBClassifier)
        self.assertEqual(step.model.get_booster().num_boosted_rounds(), 5)
        self.assertEqual(step.model.objective, "binary:logistic")

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {"no", "yes"})
        self.assertTrue(set(predictions).issubset({"no", "yes"}))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))
        np.testing.assert_array_equal(predictions, step.classes_[proba.argmax(axis=1)])
        self.assertEqual(step.score(df, y), accuracy_score(y, predictions))

    def test_fit_multiclass_returns_probabilities_for_original_labels(self) -> None:
        df = self._make_features()
        y = [
            "class_a",
            "class_b",
            "class_c",
            "class_a",
            "class_b",
            "class_c",
            "class_a",
            "class_b",
            "class_c",
        ]
        step = ActXGBoost()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        self.assertIsInstance(step.model, XGBClassifier)
        self.assertEqual(step.model.objective, "multi:softprob")
        self.assertSetEqual(set(step.classes_), {"class_a", "class_b", "class_c"})
        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 3))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        np.testing.assert_array_equal(step.predict(df), step.classes_[proba.argmax(axis=1)])

    def test_nonconsecutive_numeric_labels_are_restored(self) -> None:
        df = self._make_features()
        y = [10, 30, 50] * 3
        step = ActXGBoost()
        self._configure_fast(step)
        step.fit(self.make_dataset(df, y))
        np.testing.assert_array_equal(step.classes_, [10, 30, 50])
        self.assertTrue(set(step.predict(df)).issubset({10, 30, 50}))
        weights = np.arange(1, len(y) + 1)
        self.assertEqual(step.score(df, y, sample_weight=weights),
                         accuracy_score(y, step.predict(df), sample_weight=weights))

    def test_xgboost_configuration_reaches_the_trained_backend(self) -> None:
        step = ActXGBoost()
        params = {"max_depth": 3, "n_estimators": 7, "random_state": 11,
                  "learning_rate": 0.25, "subsample": 0.8, "min_child_weight": 0.5,
                  "colsample_bytree": 0.75}
        step.configure(params)
        step.fit(self.make_dataset(self._make_features(), y=[0, 1, 0] * 3))
        actual = step.model.get_params()
        for key, expected in params.items():
            self.assertEqual(actual[key], expected)
        self.assertEqual(actual["n_jobs"], 1)
        self.assertEqual(actual["tree_method"], "hist")
        self.assertEqual(step.model.get_booster().num_boosted_rounds(), 7)

    def test_pipeline_evaluation_and_probabilities_work_with_text_labels(self) -> None:
        X, y = make_classification_data(n_samples=30, seed=8)
        dataset = self.make_dataset(X, np.where(y == 0, "no", "yes"))
        metric = AccuracyMetric()
        candidate = Candidate(dataset, metrics=[metric], main_metric=metric)
        step = ActXGBoost()
        self._configure_fast(step)
        candidate.pipeline.set_model(step)
        scores = candidate.training_evaluate(dataset, splitter=kfold_splitter, cache_split=False)
        self.assertTrue(np.isfinite(scores[str(metric)]))
        candidate.pipeline.fit(dataset.X, dataset.y)
        np.testing.assert_array_equal(candidate.pipeline.classes_, ["no", "yes"])
        self.assertEqual(candidate.predict_proba(dataset.X).shape, (len(dataset.X), 2))
        self.assertTrue(set(candidate.predict(dataset.X)).issubset({"no", "yes"}))

    def test_suitable_target_types(self) -> None:
        df = pd.DataFrame({"f1": [0, 1, 2, 3], "f2": [1, 0, 1, 0]})
        step = ActXGBoost()

        binary = self.make_dataset(df, y=["yes", "no", "yes", "no"])
        multiclass = self.make_dataset(df, y=["a", "b", "c", "a"])
        multilabel = self.make_dataset(
            df, y=[[True, False], [False, True], [True, True], [False, False]]
        )
        continuous = self.make_dataset(df, y=[0.1, 0.2, 0.3, 0.4])

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertFalse(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))
