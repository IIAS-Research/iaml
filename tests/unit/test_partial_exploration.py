"""Generate one preprocessing choice, then replace it through real genetic mutations."""
from copy import deepcopy
import random
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from iaml.actionables.imbalance.act_random_over_sampling import ActRandomOverSampling
from iaml.actionables.imbalance.act_random_under_sampler import ActRandomUnderSampler
from iaml.actionables.normalize.act_minmax_scaler import ActMinMaxScaler
from iaml.actionables.normalize.act_robust_scaler import ActRobustScaler
from iaml.actionables.normalize.act_standard_scaler import ActStandardScaler
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml import IAML
from iaml.logger import Logger
from iaml.meta_explorer_step import MetaExplorerStep
from iaml.meta_ordered_step import MetaOrderedStep
from iaml.meta_partial_explorer_step import MetaPartialExplorerStep
from iaml.optimizers import BayesianOptimizer, GeneticOptimizer, RandomOptimizer
from iaml.shared_cache import CacheService
from iaml.void_step import VoidStep
from iaml.worker_manager import WorkerManager


class TestPartialExploration(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, Logger(), 'verbose', Logger().verbose)
        Logger().verbose = 0
        workers = WorkerManager(max_workers=1)
        self.addCleanup(setattr, workers, 'max_workers', workers.max_workers)
        workers.max_workers = 1
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        backend = CacheService()
        backend.__set_backend__({}, [], SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)
        self.X = pd.DataFrame({'a': np.arange(12, dtype=float),
                               'b': np.arange(12, dtype=float) ** 2})
        self.y = np.array([0] * 8 + [1] * 4)

    def default_stages(self, optimizer=GeneticOptimizer):
        engine = IAML(max_workers=1, max_stage_duration=1, optimizer=optimizer)
        return {step.tag: step for step in engine.first_step.steps}

    def generate(self, with_predictors=False):
        stages = self.default_stages()
        chain = MetaOrderedStep()
        for tag in ('normalize', 'imbalance', 'features_preprocessing'):
            chain.add_step(stages[tag])
        if with_predictors:
            predictors = MetaExplorerStep()
            for depth in (1, 2):
                model = ActDecisionTreeClassifier()
                model.configure('max_depth', depth)
                predictors.add_step(model)
            chain.add_step(predictors)
        for step in chain.all_steps():
            self.addCleanup(step.reset_cache)
        return chain.run(Candidate(Dataset(self.X.copy(), self.y.copy())))

    def mutate(self, candidate, tag, replacement, random_configuration=False):
        def choose(options):
            if isinstance(options[0], str):
                self.assertIn('interchange', options)
                return 'interchange'
            if isinstance(options[0], type):
                self.assertIn(replacement, options)
                return replacement
            return next(step for step in options if tag in step.tags)

        optimizer = GeneticOptimizer(nb_candidate=4)
        with patch('iaml.optimizers.genetic_optimizer.random.choice', side_effect=choose):
            if random_configuration:
                # Only replace the requested stage and leave parameter defaults intact.
                flags = [int(tag in step.tags) for step in candidate.pipeline.optimizable_step]
                with patch('iaml.optimizers.genetic_optimizer.random.getrandbits',
                           side_effect=flags):
                    return optimizer._GeneticOptimizer__random_configuration(candidate)
            return optimizer._GeneticOptimizer__mutate(candidate)

    def test_default_generation_has_one_scaler_and_no_resampling(self):
        outputs = self.generate()
        self.assertEqual(len(outputs), 1)
        candidate = outputs[0]
        steps = [step for _, step in candidate.pipeline.training_steps]
        self.assertEqual([type(step) for step in steps], [ActStandardScaler, VoidStep, VoidStep])
        self.assertEqual([step.tags for step in steps],
                         [{'normalize'}, {'imbalance'}, {'features_preprocessing'}])
        np.testing.assert_allclose(candidate.dataset.X, StandardScaler().fit_transform(self.X))
        np.testing.assert_array_equal(candidate.dataset.y, self.y)
        X, y = candidate.pipeline.fit_transform(self.X.copy(), self.y.copy())
        np.testing.assert_allclose(X, candidate.dataset.X)
        np.testing.assert_array_equal(y, self.y)

    def test_predictors_remain_explored_and_generated_pipeline_can_refit(self):
        outputs = self.generate(with_predictors=True)
        self.assertEqual(len(outputs), 2)
        self.assertEqual({item.pipeline.predictor[1].get_config('max_depth') for item in outputs},
                         {1, 2})
        for candidate in outputs:
            candidate.pipeline.fit(self.X.copy(), self.y.copy())
            self.assertEqual(candidate.predict(self.X.copy()).shape, (12,))
            self.assertEqual(len(candidate.pipeline.resamplers), 1)
            self.assertIsInstance(candidate.pipeline.resamplers[0][1], VoidStep)

    def test_optimizers_without_step_mutations_keep_initial_exploration(self):
        for optimizer in (BayesianOptimizer, RandomOptimizer, lambda **kw: GeneticOptimizer(**kw)):
            with self.subTest(optimizer=optimizer):
                stages = self.default_stages(optimizer)
                self.assertIs(type(stages['normalize']), MetaExplorerStep)
                self.assertEqual(len(stages['normalize'].steps), 5)
                self.assertTrue(stages['imbalance'].also_explore_without)
                self.assertGreater(len(stages['imbalance'].steps), 1)

    def test_genetic_subclasses_use_partial_exploration(self):
        class CustomGeneticOptimizer(GeneticOptimizer):
            pass
        stages = self.default_stages(CustomGeneticOptimizer)
        self.assertIsInstance(stages['normalize'], MetaPartialExplorerStep)
        self.assertEqual(len(stages['normalize'].steps), 1)

    def test_successive_mutations_replace_one_normalizer_without_modifying_parent(self):
        candidate = self.generate()[0]
        for replacement in (ActRobustScaler, ActMinMaxScaler, VoidStep, ActStandardScaler):
            with self.subTest(replacement=replacement):
                before = candidate.pipeline.fingerprint()
                mutated = self.mutate(candidate, 'normalize', replacement)
                self.assertIsNot(mutated, candidate)
                self.assertEqual(candidate.pipeline.fingerprint(), before)
                self.assertNotEqual(mutated.pipeline.fingerprint(), before)
                normalizers = [step for _, step in mutated.pipeline.training_steps
                               if 'normalize' in step.tags]
                self.assertEqual(len(normalizers), 1)
                self.assertIs(type(normalizers[0]), replacement)
                self.assertIn(normalizers[0], mutated.pipeline.optimizable_step)
                mutated.pipeline.fit_transform(self.X.copy(), self.y.copy())
                candidate = mutated

    def test_resampling_can_be_added_replaced_and_removed(self):
        for random_configuration in (False, True):
            candidate = self.generate()[0]
            for replacement, rows in ((ActRandomOverSampling, 16), (ActRandomUnderSampler, 8),
                                      (VoidStep, 12), (ActRandomOverSampling, 16)):
                with self.subTest(random_configuration=random_configuration, replacement=replacement):
                    candidate = self.mutate(candidate, 'imbalance', replacement, random_configuration)
                    sampler = candidate.pipeline.resamplers[0][1]
                    self.assertIs(type(sampler), replacement)
                    self.assertIn(sampler, candidate.pipeline.optimizable_step)
                    X, y = candidate.pipeline.fit_transform(self.X.copy(), self.y.copy())
                    self.assertEqual(len(X), rows)
                    self.assertEqual(len(y), rows)
                    self.assertEqual(len(candidate.pipeline.training_steps), 3)

    def test_interchangeable_does_not_override_disabled_parameter_optimization(self):
        candidate = self.generate()[0]
        step = candidate.pipeline.training_steps[0][1]
        self.assertFalse(step.optimizable)
        step.configuration = {'fixed': {'value': 3, 'description': 'Fixed parameter'}}
        optimizer = GeneticOptimizer(nb_candidate=4)
        self.assertEqual(optimizer._GeneticOptimizer__config_keys(step), [])
        with patch('iaml.optimizers.genetic_optimizer.random.getrandbits', return_value=0):
            result = optimizer._GeneticOptimizer__random_configuration(candidate)
        self.assertEqual(result.pipeline.training_steps[0][1].get_config('fixed'), 3)

    def test_predictor_cannot_be_replaced_by_a_noop(self):
        candidate = self.generate(with_predictors=True)[0]
        model = deepcopy(candidate.pipeline.predictor[1])
        with patch.object(model, 'step_with_same_tags', return_value=[type(model)]):
            self.assertIs(GeneticOptimizer._GeneticOptimizer__interchange(model), model)

    def test_generation_retains_parent_and_explores_preprocessing_choices(self):
        self.addCleanup(random.setstate, random.getstate())
        random.seed(7)
        candidate = self.generate()[0]
        candidate.computed_metrics = {candidate.main_metric: 0.75}
        original = candidate.pipeline.fingerprint()

        outputs = GeneticOptimizer(nb_candidate=4).run([candidate])

        fingerprints = {item.pipeline.fingerprint() for item in outputs}
        self.assertIn(original, fingerprints)
        self.assertGreater(len(fingerprints), 1)
        self.assertEqual(candidate.pipeline.fingerprint(), original)
        for item in outputs:
            self.assertEqual([step.tags for _, step in item.pipeline.training_steps],
                             [{'normalize'}, {'imbalance'}, {'features_preprocessing'}])


if __name__ == '__main__':
    unittest.main()
