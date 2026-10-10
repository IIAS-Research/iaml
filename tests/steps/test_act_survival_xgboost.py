"""Essential integration tests for the native XGBoost Cox predictor."""
from functools import partial
import pickle
import unittest

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from iaml.actionables.predictors.survival.act_survival_xgboost import ActSurvivalXGBoost
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.metrics import (
    BrierScoreMetric, ConcordanceIndexIPCWMetric, ConcordanceIndexMetric,
    IntegratedBrierScoreMetric,
)
from iaml.splitters import kfold_splitter


class TestActSurvivalXGBoost(unittest.TestCase):
    def setUp(self):
        marker = np.linspace(-2.0, 2.0, 48)
        order = np.random.default_rng(17).permutation(len(marker))
        self.X = pd.DataFrame({"dose [mg]<65": marker[order]})
        events = np.arange(len(marker)) % 7 != 1
        self.y = list(zip(events[order].tolist(), (np.exp(-marker[order]) + 0.2).tolist()))
        self.dataset = Dataset(self.X, self.y)

    @staticmethod
    def make_step():
        step = ActSurvivalXGBoost()
        step.configure({"n_estimators": 24, "max_depth": 2, "learning_rate": 0.2})
        return step

    def test_native_cox_backend_predicts_risk_and_scores_concordance(self):
        step = self.make_step()
        self.assertIs(step.fit(self.dataset), step)
        self.assertIsInstance(step.model, XGBRegressor)
        self.assertEqual(step.model.get_params()["objective"], "survival:cox")
        risks = step.predict(self.X)
        self.assertEqual(risks.shape, (len(self.X),))
        self.assertTrue(np.isfinite(risks).all())
        self.assertTrue((risks > 0).all())
        metric = ConcordanceIndexMetric()
        concordance = metric.compute(self.y, risks)
        self.assertGreater(concordance, 0.9)
        self.assertLess(metric.compute(self.y, -risks), 0.1)
        self.assertAlmostEqual(step.score(self.X, self.y), concordance)

    def test_cross_validation_curves_and_refit_use_the_training_baseline(self):
        metrics = [ConcordanceIndexIPCWMetric(), BrierScoreMetric(), IntegratedBrierScoreMetric()]
        step = self.make_step()
        pipeline = IAMLPipeline([("cox", step)], estimator_type="survival")
        pipeline.fit(self.X, self.y)
        candidate = Candidate(self.dataset, metrics=metrics, iaml_pipeline=pipeline)
        scores = candidate.training_evaluate(
            self.dataset, splitter=partial(kfold_splitter, nb_folds=3), cache_split=False,
        )
        self.assertEqual(set(scores), {str(metric) for metric in metrics})
        self.assertEqual(len(candidate.fold_metrics), 3)
        self.assertTrue(all(np.isfinite(score) for score in scores.values()))
        self.assertTrue(all(report["status"] == "success" for report in candidate.metric_report))

        train, test = next(kfold_splitter(self.dataset, 3))
        step.fit(train)
        np.testing.assert_array_equal(step.unique_times_, np.unique(train.y[:, 1]))
        horizon = max(time for _, time in train.y)
        times = np.linspace(0.0, horizon, 20)
        curves = step.predict_survival_function(test.X)
        survival = np.asarray([fn(times) for fn in curves])
        hazard = np.asarray([fn(times) for fn in step.predict_cumulative_hazard_function(test.X)])
        self.assertTrue(((survival >= 0) & (survival <= 1)).all())
        self.assertTrue((hazard >= 0).all())
        self.assertTrue((np.diff(survival, axis=1) <= 0).all())
        self.assertTrue((np.diff(hazard, axis=1) >= 0).all())
        np.testing.assert_allclose(survival, np.exp(-hazard))
        np.testing.assert_array_equal(survival[:, 0], 1.0)
        with self.assertRaises(ValueError):
            curves[0](np.nextafter(horizon, np.inf))

    def test_special_feature_names_pickle_and_pipeline_refit(self):
        pipeline = IAMLPipeline([("cox", self.make_step())], estimator_type="survival")
        pipeline.fit(self.X, self.y)
        restored = pickle.loads(pipeline.pickle())
        np.testing.assert_array_equal(restored.predict(self.X), pipeline.predict(self.X))
        plain = self.X.rename(columns={"dose [mg]<65": "dose"})
        reference = self.make_step().fit(Dataset(plain, self.y))
        np.testing.assert_array_equal(restored.predict(self.X), reference.predict(plain))
        with self.assertRaisesRegex(ValueError, "feature_names mismatch"):
            restored.predictor[1].predict(plain)

        times = np.linspace(0.0, max(time for _, time in self.y), 20)
        for method in ("predict_survival_function", "predict_cumulative_hazard_function"):
            expected = np.asarray([fn(times) for fn in getattr(pipeline, method)(self.X)])
            actual = np.asarray([fn(times) for fn in getattr(restored, method)(self.X)])
            np.testing.assert_array_equal(actual, expected)
        restored.fit(plain, self.y)
        np.testing.assert_array_equal(restored.predict(plain), reference.predict(plain))
        for method in ("predict_survival_function", "predict_cumulative_hazard_function"):
            expected = np.asarray([fn(times) for fn in getattr(pipeline, method)(self.X)])
            actual = np.asarray([fn(times) for fn in getattr(restored, method)(plain)])
            np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
