"""Keep survival gradient boosting unique and preserve historical imports."""
import json
import pickle
import unittest

import iaml
from iaml.actionables.predictors import survival
from iaml.actionables.predictors.survival.act_gradient_boosting_survival_analysis import (
    ActGradientBoostingSurvivalAnalysis,
)
from iaml.actionables.predictors.survival.act_survival_xgboost import (
    ActGradientBoostingSurvivalAnalysis as LegacyGradientBoostingSurvivalAnalysis,
)
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.decorators.all import find_steps_by_tag
from iaml.metastep import MetaStep
from iaml.step import Step
from tests.helpers.datasets import make_survival_data


class TestSurvivalBoostingRegistry(unittest.TestCase):
    """Import compatibility must not add duplicate choices to model exploration."""

    def test_public_and_historical_imports_share_one_class(self):
        for exported in (
            iaml.ActGradientBoostingSurvivalAnalysis,
            survival.ActGradientBoostingSurvivalAnalysis,
            LegacyGradientBoostingSurvivalAnalysis,
        ):
            with self.subTest(module=exported.__module__):
                self.assertIs(exported, ActGradientBoostingSurvivalAnalysis)

    def test_one_registered_class_preserves_all_discovery_tags(self):
        registered = [
            step_class for step_class in Step.available_steps
            if step_class.__name__ == 'ActGradientBoostingSurvivalAnalysis'
        ]
        self.assertEqual(registered, [ActGradientBoostingSurvivalAnalysis])

        for tag in ('predictor', 'tabular', 'survival', 'minimal_predictor'):
            with self.subTest(tag=tag):
                self.assertIn(ActGradientBoostingSurvivalAnalysis, find_steps_by_tag(tag))
                self.assertIn(tag, ActGradientBoostingSurvivalAnalysis().tags)

    def test_search_modes_generate_one_gradient_boosting_candidate(self):
        X, y = make_survival_data(n_samples=12, seed=42)
        dataset = Dataset(X, y)
        for tag in ('survival', 'minimal_predictor'):
            with self.subTest(tag=tag):
                stage = MetaStep(tag=tag)
                candidates = []
                for step in stage.steps:
                    if (step.__class__.__name__ == 'ActGradientBoostingSurvivalAnalysis'
                            and step.suitable(dataset)):
                        candidates.extend(step.run(Candidate(dataset)))

                self.assertEqual(len(candidates), 1)
                predictor = candidates[0].pipeline.steps[-1][1]
                self.assertIs(type(predictor), ActGradientBoostingSurvivalAnalysis)

    def test_json_pipeline_round_trip_preserves_configuration(self):
        step = LegacyGradientBoostingSurvivalAnalysis()
        step.configure({'n_estimators': 7, 'learning_rate': 0.3, 'max_depth': 2})
        step.enable = False

        payload = json.loads(json.dumps(step.json_pipeline()))
        restored = Step.from_pipeline(payload)

        self.assertIs(type(restored), ActGradientBoostingSurvivalAnalysis)
        self.assertEqual(restored.resume_configuration(), step.resume_configuration())
        self.assertFalse(restored.enable)

    def test_pickle_loads_historical_module_path(self):
        step = ActGradientBoostingSurvivalAnalysis()
        step.configure({'n_estimators': 7, 'max_depth': 2})
        payload = pickle.dumps(step, protocol=0)
        canonical_reference = (
            b'ciaml.actionables.predictors.survival.act_gradient_boosting_survival_analysis\n'
            b'ActGradientBoostingSurvivalAnalysis\n'
        )
        historical_reference = (
            b'ciaml.actionables.predictors.survival.act_survival_xgboost\n'
            b'ActGradientBoostingSurvivalAnalysis\n'
        )
        self.assertIn(canonical_reference, payload)
        # Older pickles carry this module path but the same instance state.
        legacy_payload = payload.replace(canonical_reference, historical_reference, 1)

        restored = pickle.loads(legacy_payload)

        self.assertIs(type(restored), ActGradientBoostingSurvivalAnalysis)
        self.assertEqual(restored.resume_configuration(), step.resume_configuration())


if __name__ == '__main__':
    unittest.main()
