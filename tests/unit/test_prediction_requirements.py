"""Survival forest storage follows metrics and configured plots."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from iaml.actionables.predictors.survival.act_extra_survival_trees import ActExtraSurvivalTrees
from iaml.actionables.predictors.survival.act_random_survival_forest import ActRandomSurvivalForest
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.flow import explanations
from iaml.iaml_pipeline import IAMLPipeline
from iaml.metrics import BrierScoreMetric, ConcordanceIndexIPCWMetric, IntegratedBrierScoreMetric
from iaml.plots import (CumulativeHazardModelComparisonPlot, KaplanMeierModelComparisonPlot,
                        ROCDynamiqueCurvePlot)
from iaml.study_analyses import compile_analyses


class _UnknownExplanation:
    def suitable(self, dataset):
        raise ValueError("Suitability is unavailable")


class TestPredictionRequirements(unittest.TestCase):
    def setUp(self):
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        cache.configure(None)
        self.X = pd.DataFrame({"feature": np.tile(np.arange(20, dtype=float), 2)})
        self.y = [(True, 1.0 + (index % 20) / 5) for index in range(40)]

    @staticmethod
    def pipeline(forest_class):
        step = forest_class()
        step.configure({"n_estimators": 3, "max_depth": 3, "random_state": 7})
        return IAMLPipeline([("forest", step)], estimator_type="survival")

    def test_metrics_and_plots_choose_storage_without_discarding_requested_curves(self):
        risk = ConcordanceIndexIPCWMetric()
        cases = (([risk], [], True),
                 ([risk, BrierScoreMetric()], [], False),
                 ([risk, IntegratedBrierScoreMetric()], [], False),
                 ([risk], [KaplanMeierModelComparisonPlot()], False),
                 ([risk], [CumulativeHazardModelComparisonPlot()], False),
                 ([risk], [ROCDynamiqueCurvePlot()], True),
                 ([risk], [_UnknownExplanation()], False))
        for forest_class in (ActExtraSurvivalTrees, ActRandomSurvivalForest):
            for metrics, plots, low_memory in cases:
                with self.subTest(forest=forest_class.__name__, metrics=metrics, plots=plots):
                    pipeline = self.pipeline(forest_class)
                    pipeline.fit(self.X, self.y, metrics=metrics, explanations=plots)
                    self.assertIs(pipeline.predictor[1].model.low_memory, low_memory)
                    if not low_memory:
                        self.assertEqual(len(pipeline.predict_survival_function(self.X)), 40)
                        candidate = Candidate(Dataset(self.X, self.y), metrics=metrics,
                                              iaml_pipeline=pipeline)
                        self.assertEqual(len(candidate.evaluate(self.X, self.y)), len(metrics))

    def test_cv_propagates_metrics_and_refit_can_explicitly_keep_future_curves(self):
        def split(dataset):
            yield Dataset(self.X.iloc[:20], self.y[:20]), Dataset(self.X.iloc[20:], self.y[20:])

        for forest_class in (ActExtraSurvivalTrees, ActRandomSurvivalForest):
            with self.subTest(forest=forest_class.__name__):
                candidate = Candidate(Dataset(self.X, self.y), metrics=[ConcordanceIndexIPCWMetric()],
                                      iaml_pipeline=self.pipeline(forest_class))
                fitted_modes = []
                original_fit = forest_class.fit

                def fit(step, dataset):
                    result = original_fit(step, dataset)
                    fitted_modes.append(step.model.low_memory)
                    return result

                with patch.object(forest_class, "fit", fit):
                    self.assertTrue(candidate.training_evaluate(candidate.dataset, splitter=split,
                                                               cache_split=False))
                    candidate.explanation_definitions = compile_analyses(
                        explanations(KaplanMeierModelComparisonPlot))
                    self.assertTrue(candidate.training_evaluate(candidate.dataset, splitter=split,
                                                               cache_split=False))
                self.assertEqual(fitted_modes, [True, False])
                candidate.pipeline.predictor[1].configure({"low_memory": False})
                candidate.pipeline.fit(self.X, self.y, metrics=candidate.metrics)
                self.assertEqual(len(candidate.pipeline.predict_survival_function(self.X)), 40)
