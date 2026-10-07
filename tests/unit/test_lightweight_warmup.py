"""Warmup uses declared lightweight models and evaluations report their timing."""
import unittest
from unittest.mock import patch

from iaml.actionables.normalize.act_standard_scaler import ActStandardScaler
from iaml.actionables.predictors.classifier.act_logistic_regression import ActLogisticRegression
from iaml.actionables.predictors.classifier.act_svm_svc import ActSVMSVC
from iaml.candidate import Candidate
from iaml.iaml import IAML, process_executor
from iaml.logger import Logger
from tests.unit import test_sliding_stage_search as sliding


class LightweightWarmupTests(unittest.TestCase):
    setUp = sliding.SlidingStageSearchTests.setUp
    candidate = sliding.SlidingStageSearchTests.candidate
    evaluate = sliding.SlidingStageSearchTests.evaluate

    def test_warmup_prefers_declared_fast_models_and_simpler_preprocessing(self):
        slow = self.candidate('slow', 2)
        slow.pipeline.set_model(ActSVMSVC())
        prepared = self.candidate('prepared', 2)
        prepared.pipeline.add_transform(ActStandardScaler())
        prepared.pipeline.set_model(ActLogisticRegression())
        simple = self.candidate('simple', 2)
        simple.pipeline.set_model(ActLogisticRegression())
        candidates = [slow, prepared, simple]
        fingerprints = [candidate.pipeline.fingerprint() for candidate in candidates]

        selected = min(candidates, key=IAML._IAML__warmup_priority)

        self.assertIs(selected, simple)
        self.assertEqual([candidate.pipeline.fingerprint() for candidate in candidates], fingerprints)
        self.assertIs(min([slow], key=IAML._IAML__warmup_priority), slow)

    def test_queue_logs_population_and_keeps_submission_timestamp(self):
        Logger().verbose = 2
        self.engine.executor.completions = [(1, {'model': 0.8})]
        with patch.object(Logger(), 'info') as log:
            self.evaluate([self.candidate('model', 2)])
        messages = [str(call.args[0]) for call in log.call_args_list]
        self.assertTrue(any('queued on 12 rows' in message for message in messages))
        self.assertEqual(self.engine.executor.submissions[0][2]['queued_at'], 0.0)

    def test_worker_reports_wait_and_execution_separately(self):
        Logger().verbose = 2
        candidate = self.candidate('model', 2)

        def evaluate(current, *args, **kwargs):
            current.computed_metrics = {current.main_metric: 0.8}

        with (patch.object(Candidate, 'training_evaluate', evaluate),
              patch.object(Logger(), 'info') as log,
              patch('iaml.iaml.time.monotonic', side_effect=[12.0, 14.0])):
            result = process_executor(candidate, self.dataset, evaluation_id=7, queued_at=3.0)
        self.assertEqual(result.evaluation_id, 7)
        messages = [str(call.args[0]) for call in log.call_args_list]
        self.assertTrue(any('started after 9.00s waiting' in message for message in messages))
        self.assertTrue(any('finished in 2.00s (success)' in message for message in messages))

    def test_worker_reports_failure_without_swallowing_it(self):
        Logger().verbose = 2
        candidate = self.candidate('model', 2)
        with (patch.object(Candidate, 'training_evaluate', side_effect=ValueError('failed')),
              patch.object(Logger(), 'info') as log,
              patch('iaml.iaml.time.monotonic', side_effect=[12.0, 14.0])):
            with self.assertRaisesRegex(ValueError, 'failed'):
                process_executor(candidate, self.dataset, evaluation_id=7, queued_at=3.0)
        self.assertIn('(failed)', log.call_args_list[-1].args[0])


if __name__ == '__main__':
    unittest.main()
