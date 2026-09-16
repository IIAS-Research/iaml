"""Candidate ranking and optimization must respect each metric's direction."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metrics import (
    AccuracyMetric,
    BrierScoreMetric,
    ClassificationErrorMetric,
    IntegratedBrierScoreLossMetric,
    IntegratedBrierScoreMetric,
    MeanAbsoluteErrorMetric,
    MeanSquaredErrorMetric,
    MeanSquaredLogErrorMetric,
    MedianAbsoluteErrorMetric,
    R2ScoreMetric,
)
from iaml.optimizers.optimizer import Optimizer


LOSSES = (
    ClassificationErrorMetric,
    MeanAbsoluteErrorMetric,
    MeanSquaredErrorMetric,
    MeanSquaredLogErrorMetric,
    MedianAbsoluteErrorMetric,
    BrierScoreMetric,
    IntegratedBrierScoreMetric,
)


class TestMetricDirection(unittest.TestCase):
    def setUp(self):
        self.previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", self.previous_verbose)
        self.dataset = Dataset(
            pd.DataFrame({"feature": np.arange(24, dtype=float) + 0.5}),
            np.linspace(0.1, 2.4, 24),
        )

    def make_candidate(self, metric, value, use_name=False, include_metrics=True):
        candidate = Candidate(
            self.dataset,
            metrics=[metric] if include_metrics else None,
            main_metric=str(metric) if use_name else metric,
            iaml_pipeline=IAMLPipeline(estimator_type="regressor"),
        )
        candidate.computed_metrics = {str(metric): value}
        return candidate

    def test_all_losses_rank_lower_values_first_without_changing_raw_values(self):
        for metric_class in LOSSES:
            for use_name in (False, True):
                with self.subTest(metric=metric_class.__name__, use_name=use_name):
                    metric = metric_class()
                    best = self.make_candidate(metric, 0.0, use_name=use_name)
                    middle = self.make_candidate(metric, 0.48, use_name=use_name)
                    worst = self.make_candidate(metric, 77.04, use_name=use_name)

                    ranked = sorted([middle, worst, best], reverse=True)

                    self.assertIs(ranked[0], best)
                    self.assertIs(ranked[-1], worst)
                    self.assertGreater(best, middle)
                    self.assertLess(worst, middle)
                    self.assertEqual(middle.get_main_metric_value(), 0.48)
                    self.assertEqual(middle.get_main_metric_score(), -0.48)
                    self.assertEqual(middle.computed_metrics, {str(metric): 0.48})

    def test_maximized_metrics_still_rank_higher_values_first(self):
        for metric, best_value, worse_value in (
            (AccuracyMetric(), 0.9, 0.2),
            (R2ScoreMetric(), -0.5, -4.0),
            (IntegratedBrierScoreLossMetric(), 0.8, 0.3),
        ):
            with self.subTest(metric=str(metric)):
                best = self.make_candidate(metric, best_value)
                worse = self.make_candidate(metric, worse_value)

                self.assertIs(sorted([worse, best], reverse=True)[0], best)
                self.assertGreater(best, worse)
                self.assertEqual(best.get_main_metric_score(), best_value)
                self.assertEqual(best.get_main_metric_value(), best_value)

    def test_metric_instance_direction_survives_copies_without_a_metrics_list(self):
        # Configure an existing metric instance rather than register a test-only
        # subclass that IAML would discover as a production metric.
        custom_metric = R2ScoreMetric()
        custom_metric.greater_is_better = False
        for metric in (MeanSquaredErrorMetric(), custom_metric):
            for include_metrics in (False, True):
                with self.subTest(metric=str(metric), include_metrics=include_metrics):
                    original = self.make_candidate(
                        metric, 0.25, include_metrics=include_metrics
                    )
                    worse = self.make_candidate(metric, 0.75)
                    for candidate in (
                        original,
                        original.to_output(),
                        original.to_input(),
                        original.to_output().to_input(),
                    ):
                        candidate.computed_metrics = {str(metric): 0.25}
                        self.assertGreater(candidate, worse)
                        self.assertEqual(candidate.get_main_metric_value(), 0.25)
                        self.assertEqual(candidate.get_main_metric_score(), -0.25)

    def test_missing_main_metric_cannot_outrank_a_valid_loss(self):
        metric = MeanSquaredErrorMetric()
        valid = self.make_candidate(metric, 77.04)
        missing = self.make_candidate(metric, 0.0)
        missing.computed_metrics = {"r2_score": 0.9}
        unevaluated = self.make_candidate(metric, 0.0)
        unevaluated.computed_metrics = {}

        self.assertGreater(valid, missing)
        self.assertGreater(valid, unevaluated)
        self.assertIs(sorted([missing, unevaluated, valid], reverse=True)[0], valid)

    def test_evaluated_regressors_are_ranked_by_smallest_mse(self):
        metric = MeanSquaredErrorMetric()
        candidates = []
        for model in (
            DummyRegressor(strategy="constant", constant=10.0),
            DummyRegressor(strategy="mean"),
        ):
            model.fit(self.dataset.X, self.dataset.y)
            candidate = Candidate(
                self.dataset,
                metrics=[metric],
                main_metric=metric,
                iaml_pipeline=IAMLPipeline(
                    [("regressor", model)], estimator_type="regressor"
                ),
            )
            candidate.computed_metrics = candidate.evaluate(
                self.dataset.X, self.dataset.y
            )
            candidates.append(candidate)

        worse, best = candidates
        self.assertAlmostEqual(worse.get_main_metric_value(), 77.04166666666667)
        self.assertAlmostEqual(best.get_main_metric_value(), 0.47916666666667)
        self.assertIs(sorted(candidates, reverse=True)[0], best)

    def test_optimization_patience_resets_when_loss_decreases(self):
        for metric, values in (
            (MeanSquaredErrorMetric(), [4.0, 3.0, 2.0, 2.0]),
            (R2ScoreMetric(), [-4.0, -3.0, -2.0, -2.0]),
        ):
            with self.subTest(metric=str(metric)):
                candidates = [self.make_candidate(metric, value) for value in values]
                optimizer = Mock(spec=Optimizer)
                optimizer.finished = False
                optimizer.run.side_effect = lambda previous: previous
                iaml = IAML(max_workers=1, max_duration=30, max_stage_duration=30)

                with patch.object(
                    iaml,
                    "_IAML__run_evaluations",
                    side_effect=[[candidate] for candidate in candidates[1:]],
                ) as evaluate:
                    results = iaml._IAML__optimize(
                        self.dataset,
                        candidates[:1],
                        optimizer=optimizer,
                        patience=1,
                        max_duration=30,
                    )

                self.assertEqual(evaluate.call_count, 3)
                self.assertEqual(optimizer.run.call_count, 3)
                self.assertIs(results[0], candidates[-1])
                self.assertEqual(results[0].get_main_metric_value(), values[-1])
