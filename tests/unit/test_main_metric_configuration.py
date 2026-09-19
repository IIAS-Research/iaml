"""Use the requested metric configuration in IAML's evaluation metrics."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.metrics import AccuracyMetric, F1ScoreMetric, PrecisionMetric, RecallMetric


class TestMainMetricConfiguration(unittest.TestCase):
    def setUp(self):
        previous_verbose = Logger().verbose
        Logger().verbose = 0
        self.addCleanup(setattr, Logger(), "verbose", previous_verbose)
        self.dataset = Dataset(
            pd.DataFrame({"feature": np.arange(24, dtype=float) + 0.5}),
            np.tile([0, 0, 1], 8),
        )

    def select_metrics(self, main_metric=None):
        engine = IAML(main_metric=main_metric, max_workers=1, max_stage_duration=1)
        return engine._IAML__metrics_selection(
            self.dataset.X, self.dataset.y, self.dataset.type_of_target,
        )

    def test_configured_metrics_are_used_for_actual_candidate_evaluation(self):
        model = DummyClassifier(strategy="constant", constant=1)
        model.fit(self.dataset.X, self.dataset.y)
        for metric_class, expected_positive_one in (
            (PrecisionMetric, 1 / 3),
            (RecallMetric, 1.0),
            (F1ScoreMetric, 0.5),
        ):
            for pos_label, expected in ((0, 0.0), (1, expected_positive_one)):
                with self.subTest(metric=metric_class.__name__, pos_label=pos_label):
                    requested = metric_class(pos_label=pos_label)
                    metrics = self.select_metrics(requested)
                    matching = [metric for metric in metrics if str(metric) == str(requested)]
                    self.assertEqual(len(matching), 1)
                    self.assertIs(matching[0], requested)

                    candidate = Candidate(
                        self.dataset,
                        metrics=metrics,
                        main_metric=requested,
                        iaml_pipeline=IAMLPipeline(
                            [("classifier", model)], estimator_type="classifier",
                        ),
                    )
                    for evaluated in (candidate, candidate.to_output(), candidate.to_input()):
                        scores = evaluated.evaluate(self.dataset.X, self.dataset.y)
                        self.assertAlmostEqual(scores[str(requested)], expected)

    def test_supplied_metric_is_not_reconstructed(self):
        requested = PrecisionMetric(pos_label=0)
        with patch.object(PrecisionMetric, "__init__", side_effect=AssertionError(
            "The configured metric must not be constructed again"
        )):
            metrics = self.select_metrics(requested)

        self.assertIn(requested, metrics)
        self.assertEqual(requested.pos_label, 0)

    def test_automatic_metrics_keep_their_default_configuration(self):
        metrics = self.select_metrics()
        for metric_class in (PrecisionMetric, RecallMetric, F1ScoreMetric):
            matching = [metric for metric in metrics if type(metric) is metric_class]
            self.assertEqual(len(matching), 1)
            self.assertIsNone(matching[0].pos_label)

    def test_explicit_main_metric_is_kept_despite_automatic_selection_filter(self):
        self.dataset = Dataset(self.dataset.X, np.tile([0, 1], 12))
        requested = AccuracyMetric()
        self.assertFalse(requested.suitable(
            self.dataset.X, self.dataset.y, self.dataset.type_of_target,
        ))
        self.assertIn(requested, self.select_metrics(requested))
        self.assertFalse(any(isinstance(metric, AccuracyMetric) for metric in self.select_metrics()))
