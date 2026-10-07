"""Candidates compared after downsizing must share an evaluation population."""
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from tests.unit import test_sliding_stage_search as sliding
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.iaml import IAML
from iaml.meta_singleton import MetaSingleton
from iaml.metrics.accuracy_metric import AccuracyMetric


class PopulationComparisonTests(unittest.TestCase):
    setUp = sliding.SlidingStageSearchTests.setUp
    candidate = sliding.SlidingStageSearchTests.candidate
    evaluate = sliding.SlidingStageSearchTests.evaluate

    def test_late_original_score_is_retained_but_not_ranked_on_sample(self):
        slow, sampled = self.candidate('slow', 2), self.candidate('sampled', 3)
        subset = self.dataset.sample(6)
        executor = self.engine.executor
        executor.completions = [(1, {}), (1, {'slow': 0.99, 'sampled': 0.7})]

        self.assertEqual(self.evaluate([slow]), [])
        current = self.evaluate([sampled], dataset=subset)

        self.assertEqual([candidate.pipeline.predictor[0] for candidate in current], ['sampled'])
        self.assertEqual(len(self.engine._completed_evaluations), 2)
        self.assertEqual(len(self.engine.training_history), 2)
        chosen = self.engine._IAML__compare_completed_candidates(current, subset, timeout=0)
        self.assertEqual([candidate.pipeline.predictor[0] for candidate in chosen], ['sampled'])
        self.assertEqual(len(executor.submissions), 2)

    def test_earlier_candidate_is_rescored_on_current_data_without_duplication(self):
        slow, sampled = self.candidate('slow', 2), self.candidate('sampled', 3)
        subset = self.dataset.sample(6)
        executor = self.engine.executor
        executor.completions = [
            (1, {}), (1, {'slow': 0.99, 'sampled': 0.8}), (1, {'slow': 0.7}),
        ]
        self.evaluate([slow])
        current = self.evaluate([sampled], dataset=subset)

        compared = self.engine._IAML__compare_completed_candidates(current, subset, timeout=4)

        self.assertEqual([candidate.pipeline.predictor[0] for candidate in compared], ['sampled', 'slow'])
        self.assertEqual([candidate.get_main_metric_value() for candidate in compared], [0.8, 0.7])
        self.assertEqual([len(job[1].X) for job in executor.submissions], [12, 6, 6])
        self.assertEqual(executor.waits[-1], (4, True))
        self.assertEqual(len(self.engine._completed_evaluations), 3)
        self.assertEqual(len(self.engine.training_history), 3)
        repeated = self.engine._IAML__compare_completed_candidates(compared, subset, timeout=4)
        self.assertEqual(len(repeated), 2)
        self.assertEqual(len(executor.submissions), 3)

    def test_global_deadline_stops_comparison_and_keeps_comparable_results(self):
        slow, sampled = self.candidate('slow', 2), self.candidate('sampled', 3)
        subset = self.dataset.sample(6)
        executor = self.engine.executor
        executor.completions = [
            (1, {'slow': 0.99}), (1, {'sampled': 0.8}), (100, {}),
        ]
        self.evaluate([slow])
        current = self.evaluate([sampled], dataset=subset)
        self.engine._search_deadline = 2.5

        compared = self.engine._IAML__compare_completed_candidates(current, subset, timeout=0.5)

        self.assertEqual(self.now, 2.5)
        self.assertEqual(executor.waits[-1], (0.5, True))
        self.assertEqual(executor.pending, [])
        self.assertEqual([candidate.pipeline.predictor[0] for candidate in compared], ['sampled'])
        self.assertFalse(self.engine._IAML__evaluations_pending())
        self.assertEqual(len(self.engine._completed_evaluations), 2)

    def test_only_earlier_valid_population_can_supply_fallback_at_deadline(self):
        slow = self.candidate('slow', 2)
        subset = self.dataset.sample(6)
        executor = self.engine.executor
        executor.completions = [(1, {}), (1, {'slow': 0.9})]
        self.evaluate([slow])
        self.assertEqual(self.evaluate([], dataset=subset), [])
        self.engine._search_deadline = self.now

        fallback, population = self.engine._IAML__previous_population(subset)
        compared = self.engine._IAML__compare_completed_candidates(fallback, population, timeout=0)

        self.assertIs(population, self.dataset)
        self.assertEqual([candidate.pipeline.predictor[0] for candidate in compared], ['slow'])
        self.assertEqual(compared[0].get_main_metric_value(), 0.9)
        self.assertEqual(len(executor.submissions), 1)

    def test_fallback_prefers_current_population_without_comparing_other_scores(self):
        original, sampled = self.candidate('original', 2), self.candidate('sampled', 3)
        subset = self.dataset.sample(6)
        executor = self.engine.executor
        executor.completions = [(1, {'original': 0.99}), (1, {'sampled': 0.6})]
        self.evaluate([original])
        self.evaluate([sampled], dataset=subset)

        fallback, population = self.engine._IAML__previous_population(subset)

        self.assertIs(population, subset)
        self.assertEqual([candidate.pipeline.predictor[0] for candidate in fallback], ['sampled'])

    def test_fit_can_refit_an_old_result_when_sample_has_no_score_at_deadline(self):
        self.addCleanup(setattr, MetaSingleton, '_instances', dict(MetaSingleton._instances))
        X = pd.DataFrame({'row_number': np.arange(520, dtype=float)})
        y = np.tile([False, True], 260)
        with patch.object(IAML, 'default_pipeline'):
            engine = IAML(
                max_workers=1, max_duration=6, max_stage_duration=1,
                time_before_sample_use=2, main_metric=AccuracyMetric(),
                splitter=self.engine.splitter,
            )
        engine.first_step = ActDecisionTreeClassifier()
        engine.minimal_predictor_step = None

        class LateOriginalExecutor(sliding.StageExecutor):
            def join(executor, timeout, *, cancel_pending=None):
                if len(executor.waits) == 2:
                    self.now = 6
                return super().join(timeout, cancel_pending=cancel_pending)

        executor = LateOriginalExecutor(self)
        executor.shutdown = Mock()
        # Only the original job enters the pool; sampled work stays deferred.
        executor.accept_limit = 1
        executor.completions = [
            (1, {}), (1, {}), (0, {ActDecisionTreeClassifier.name: 0.9}), (0, {}),
        ]
        with (
            patch('iaml.iaml.TimedPoolExecutor', return_value=executor),
            patch.object(engine, '_IAML__metrics_selection', return_value=[AccuracyMetric()]),
            patch.object(engine, '_IAML__optimize', side_effect=lambda ds, cands, **kw: cands),
        ):
            fitted = engine.fit(X, y, generation_sample_size=12, verbose=0)

        self.assertEqual(self.now, 6)
        self.assertEqual(len(fitted), 1)
        self.assertEqual(fitted[0].get_main_metric_value(), 0.9)
        self.assertEqual(fitted[0].pipeline.predictor[1].model.tree_.n_node_samples[0], 520)
        self.assertEqual(len(executor.submissions), 1)


if __name__ == '__main__':
    unittest.main()
