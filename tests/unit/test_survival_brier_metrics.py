"""Regression tests for Brier evaluation on fractional follow-up times."""
import unittest

import numpy as np
import pandas as pd
from sksurv.functions import StepFunction
from sksurv.metrics import brier_score, integrated_brier_score

from iaml.actionables.predictors.survival.act_survival_tree import ActSurvivalTree
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.metrics import (
    BrierScoreMetric,
    IntegratedBrierScoreLossMetric,
    IntegratedBrierScoreMetric,
)


SURVIVAL_DTYPE = [("event", "bool"), ("time", "float")]
METRICS_AND_EXPECTED = (
    (BrierScoreMetric, 0.25),
    (IntegratedBrierScoreMetric, 0.25),
    (IntegratedBrierScoreLossMetric, 0.75),
)


class TestSurvivalBrierMetrics(unittest.TestCase):
    """Use the actual survival functions and sksurv scoring implementation."""

    @staticmethod
    def constant_predictions(times):
        return [
            StepFunction(np.asarray(times), np.full(len(times), 0.5))
            for _ in times
        ]

    def test_fractional_durations_have_the_expected_score_in_supported_formats(self):
        samples = [(True, 0.1), (True, 0.2), (True, 0.3), (False, 0.4)]
        predictions = self.constant_predictions([sample[1] for sample in samples])
        targets = (
            samples,
            np.array(samples, dtype=SURVIVAL_DTYPE),
            pd.DataFrame(samples, columns=["event", "time"]),
        )

        # With survival probability 0.5, every squared error is exactly 0.25.
        # Censoring occurs only at the excluded upper endpoint.
        for target in targets:
            for metric_class, expected in METRICS_AND_EXPECTED:
                with self.subTest(target=type(target).__name__, metric=metric_class.__name__):
                    actual = metric_class().compute(target, predictions, target)
                    self.assertAlmostEqual(actual, expected)

    def test_changing_time_units_or_origin_preserves_scores(self):
        base_times = np.array([0.1, 0.2, 0.3, 0.4])
        probabilities = np.array([0.9, 0.7, 0.4, 0.2])
        base_samples = list(zip([True, True, True, False], base_times))
        base_predictions = [StepFunction(base_times, probabilities) for _ in base_samples]

        for metric_class, _ in METRICS_AND_EXPECTED:
            with self.subTest(metric=metric_class.__name__):
                baseline = metric_class().compute(base_samples, base_predictions, base_samples)
                self.assertTrue(np.isfinite(baseline))
                for scale, offset in ((10, 0), (100, 0), (1, 10)):
                    with self.subTest(scale=scale, offset=offset):
                        times = base_times * scale + offset
                        samples = list(zip([True, True, True, False], times))
                        predictions = [StepFunction(times, probabilities) for _ in samples]
                        score = metric_class().compute(samples, predictions, samples)
                        self.assertAlmostEqual(score, baseline)

    def test_survival_tree_predictions_match_sksurv_on_the_documented_grid(self):
        train_X = pd.DataFrame({"feature": np.arange(8, dtype=float)})
        train_y = [
            (True, 0.1), (True, 0.2), (False, 0.3), (True, 0.4),
            (True, 0.5), (False, 0.6), (True, 0.7), (False, 0.8),
        ]
        step = ActSurvivalTree()
        step.configure({"min_samples_leaf": 1, "max_depth": 3, "random_state": 0})
        step.fit(Dataset(train_X, train_y))
        test_X = pd.DataFrame({"feature": [0.5, 2.5, 4.5, 6.5]})
        predictions = step.predict_survival_function(test_X)
        train_struct = np.array(train_y, dtype=SURVIVAL_DTYPE)
        candidate = Candidate(
            Dataset(train_X, train_y),
            metrics=[metric_class() for metric_class, _ in METRICS_AND_EXPECTED],
            iaml_pipeline=IAMLPipeline([("survival", step)], estimator_type="survival"),
        )

        for last_time in (0.75, 0.95):
            with self.subTest(last_time=last_time):
                test_y = [(True, 0.15), (False, 0.35), (True, 0.55), (True, last_time)]
                expected_y = test_y.copy()
                if last_time >= 0.8:
                    expected_y[-1] = (False, 0.8)
                test_struct = np.array(expected_y, dtype=SURVIVAL_DTYPE)
                # The grid contains 100 evenly spaced points; sksurv excludes
                # the upper follow-up boundary from its evaluation domain.
                times = np.linspace(0.15, expected_y[-1][1], 100, endpoint=False)
                probabilities = np.asarray([[fn(time) for time in times] for fn in predictions])
                expected_brier = brier_score(
                    train_struct, test_struct, probabilities[:, -1], times[-1]
                )[1][0]
                expected_ibs = integrated_brier_score(
                    train_struct, test_struct, probabilities, times
                )

                self.assertAlmostEqual(
                    BrierScoreMetric().compute(test_y, predictions, train_y), expected_brier
                )
                self.assertAlmostEqual(
                    IntegratedBrierScoreMetric().compute(test_y, predictions, train_y),
                    expected_ibs,
                )
                self.assertAlmostEqual(
                    IntegratedBrierScoreLossMetric().compute(test_y, predictions, train_y),
                    1 - expected_ibs,
                )
                computed = candidate.evaluate(test_X, test_y)
                expected = {
                    "brier_score": expected_brier,
                    "integrated_brier_score": expected_ibs,
                    "integrated_brier_score_loss": 1 - expected_ibs,
                }
                self.assertEqual(computed.keys(), expected.keys())
                for name, value in expected.items():
                    self.assertAlmostEqual(computed[name], value)

    def test_identical_follow_up_times_are_rejected(self):
        samples = [(True, 0.2), (False, 0.2)]
        train = [(True, 0.1), (False, 0.4)]
        predictions = self.constant_predictions([0.1, 0.4])

        for metric_class, _ in METRICS_AND_EXPECTED:
            with self.subTest(metric=metric_class.__name__):
                with self.assertRaises(ValueError):
                    metric_class().compute(samples, predictions, train)

    def test_brier_accepts_the_only_representable_evaluation_time(self):
        upper = np.nextafter(1.0, np.inf)
        samples = [(True, 1.0), (False, upper)]
        predictions = self.constant_predictions([1.0, upper])

        self.assertAlmostEqual(BrierScoreMetric().compute(samples, predictions, samples), 0.25)

    def test_integrated_brier_rejects_a_single_representable_evaluation_time(self):
        upper = np.nextafter(1.0, np.inf)
        samples = [(True, 1.0), (False, upper)]
        predictions = self.constant_predictions([1.0, upper])

        for metric_class in (IntegratedBrierScoreMetric, IntegratedBrierScoreLossMetric):
            with self.subTest(metric=metric_class.__name__):
                with self.assertRaisesRegex(ValueError, "[Tt]wo|distinct"):
                    metric_class().compute(samples, predictions, samples)


if __name__ == "__main__":
    unittest.main()
