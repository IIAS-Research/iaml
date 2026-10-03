"""Compiled search policies survive optimization and fitted cache reuse."""
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.flow import Const, Int, use
from iaml.iaml_pipeline import IAMLPipeline
from iaml.logger import Logger
from iaml.optimizers import BayesianOptimizer, GeneticOptimizer, RandomOptimizer
from iaml.search_policy import parameter_keys
from iaml.shared_cache import CacheService
from iaml.step import Step
from iaml.steps import DecisionTreeClassifier
from iaml.void_step import VoidStep


class _SearchPredictor(Step):
    def __init__(self):
        super().__init__()
        self.optimizable = True
        self.tags = {'predictor'}
        self.configuration = {
            'depth': {'value': 3, 'range': [-5, 10]},
            'locked': {'value': 4, 'range': [1, 10]},
        }

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class _SearchResampler(_SearchPredictor):
    predict = None

    def __init__(self):
        super().__init__()
        self.tags = {'imbalance'}

    def resample(self, X, y):
        return X, y


class _SearchTransform(_SearchResampler):
    fit_calls = 0
    resample = None

    def __init__(self):
        super().__init__()
        self.tags = {'normalize'}
        self.mean = None

    def fit(self, dataset):
        type(self).fit_calls += 1
        self.mean = float(np.mean(dataset.y))
        return self

    def transform(self, X):
        return X + self.mean


def compiled(step, variant='first', alias=None, fixed=('locked',)):
    step._flow_explicit = True
    step._flow_node_id = variant
    step._flow_variant_id = variant
    step._flow_alias = alias
    step._flow_choice_id = 'slot'
    step._flow_parameters = {
        key: {'fixed': key in fixed, 'optimizable': key not in fixed,
              'domain': list(config['range'])}
        for key, config in step.configuration.items()
    }
    return step


class TestFlowOptimizers(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, Logger(), 'verbose', Logger().verbose)
        Logger().verbose = 0
        self.dataset = Dataset(pd.DataFrame({'x': np.arange(12, dtype=float)}), [0, 1] * 6)

    def candidate(self, predictor=None, resampler=None):
        pipeline = IAMLPipeline(estimator_type='classifier')
        if resampler is not None:
            pipeline.add_resample(resampler)
        pipeline.set_model(predictor if predictor is not None else compiled(_SearchPredictor()))
        candidate = Candidate(self.dataset.copy(), iaml_pipeline=pipeline)
        candidate.computed_metrics = {candidate.main_metric: 0.8}
        return candidate

    def test_all_optimizers_keep_const_and_original_candidate(self):
        for optimizer in (RandomOptimizer(), BayesianOptimizer(), GeneticOptimizer(nb_candidate=4)):
            with self.subTest(optimizer=type(optimizer).__name__):
                candidate = self.candidate()
                before = candidate.pipeline.fingerprint()
                outputs = optimizer.run([candidate])
                self.assertTrue(outputs)
                self.assertEqual(candidate.pipeline.fingerprint(), before)
                for item in outputs:
                    self.assertEqual(item.pipeline.predictor[1].get_config('locked'), 4)

    def test_explicit_domain_activates_only_its_parameter(self):
        step = compiled(_SearchPredictor())
        step.optimizable = False
        step._flow_parameters['locked']['fixed'] = False
        step._flow_parameters['locked']['optimizable'] = False
        self.assertEqual(parameter_keys(step), ['depth'])

    def seed_candidate(self, explicit_domain):
        recipe = use(DecisionTreeClassifier)
        recipe.configure(**{key: Const(param.value) for key, param in recipe.parameters.items()
                            if key != 'random_state'})
        if explicit_domain:
            recipe.configure(random_state=Int(1, 5, initial=3))
        return self.candidate(recipe.instantiate())

    def test_explicit_domain_overrides_ignored_seed_but_default_seed_stays_ignored(self):
        default = self.seed_candidate(False)
        explicit = self.seed_candidate(True)
        for optimizer in (GeneticOptimizer(nb_candidate=4), BayesianOptimizer()):
            with self.subTest(optimizer=type(optimizer).__name__):
                self.assertEqual(parameter_keys(default.pipeline.predictor[1],
                                                optimizer.ignored_configs), [])
                self.assertEqual(parameter_keys(explicit.pipeline.predictor[1],
                                                optimizer.ignored_configs), ['random_state'])
                default_outputs = optimizer.run([deepcopy(default)])
                self.assertTrue(default_outputs)
                self.assertEqual({item.pipeline.predictor[1].get_config('random_state')
                                  for item in default_outputs}, {42})

    def test_genetic_mutates_an_explicit_seed_domain(self):
        candidate = self.seed_candidate(True)
        with patch('random.getrandbits', return_value=0), \
                patch('random.uniform', return_value=4):
            outputs = GeneticOptimizer(nb_candidate=4).run([candidate])
        self.assertEqual({item.pipeline.predictor[1].get_config('random_state')
                          for item in outputs}, {3, 4})
        self.assertEqual(candidate.pipeline.predictor[1].get_config('random_state'), 3)

    def test_legacy_seed_remains_ignored_by_genetic_and_bayesian(self):
        step = _SearchPredictor()
        step.configuration['random_state'] = {'value': 42}
        candidate = self.candidate(step)
        for optimizer in (GeneticOptimizer(nb_candidate=4), BayesianOptimizer()):
            with self.subTest(optimizer=type(optimizer).__name__):
                self.assertNotIn('random_state', parameter_keys(step, optimizer.ignored_configs))
                outputs = optimizer.run([deepcopy(candidate)])
                self.assertTrue(outputs)
                self.assertEqual({item.pipeline.predictor[1].get_config('random_state')
                                  for item in outputs}, {42})

    def test_bayesian_exports_and_applies_an_explicit_seed_domain(self):
        candidate = self.seed_candidate(True)
        optimizer = BayesianOptimizer()
        optimizer._initialize_search_space([candidate])
        backend = next(iter(optimizer.skopt_optimizers.values()))
        self.assertEqual(backend['param_keys'], ['0_random_state'])
        self.assertEqual(backend['dimensions'][0].bounds, (1, 5))
        self.assertEqual(optimizer._export_params(candidate, backend), [3])
        with patch.object(backend['optimizer'], 'ask', return_value=[[4]]):
            outputs = optimizer.run([candidate])
        self.assertEqual([item.pipeline.predictor[1].get_config('random_state')
                          for item in outputs], [3, 4])
        self.assertEqual(candidate.pipeline.predictor[1].get_config('random_state'), 3)

    def test_const_does_not_clamp_to_inactive_domain(self):
        step = compiled(_SearchPredictor())
        step.configure('locked', 30)
        self.assertTrue(step.check_configuration())
        self.assertEqual(step.get_config('locked'), 30)

    def test_bayesian_bounds_keep_zero_and_negative_integers(self):
        optimizer = BayesianOptimizer()
        optimizer._initialize_search_space([self.candidate()])
        backend = next(iter(optimizer.skopt_optimizers.values()))
        self.assertEqual(backend['param_keys'], ['0_depth'])
        self.assertEqual(backend['dimensions'][0].bounds, (-5, 10))

    def test_bayesian_spaces_distinguish_domain_fixed_and_class(self):
        optimizer = BayesianOptimizer()
        candidate = self.candidate()
        original = optimizer._generate_structure_id(candidate)
        same = deepcopy(candidate)
        same.pipeline.predictor[1].configure('depth', 8)
        self.assertEqual(original, optimizer._generate_structure_id(same))
        same.pipeline.predictor[1]._flow_alias = 'renamed'
        same.pipeline.predictor[1]._flow_node_id = 'another-id'
        self.assertEqual(original, optimizer._generate_structure_id(same))
        same.pipeline.predictor[1].configure('locked', 5)
        self.assertNotEqual(original, optimizer._generate_structure_id(same))
        other = deepcopy(candidate)
        step = other.pipeline.predictor[1]
        step.configuration['depth']['range'] = [0, 20]
        step._flow_parameters['depth']['domain'] = [0, 20]
        self.assertNotEqual(original, optimizer._generate_structure_id(other))
        class _OtherPredictor(_SearchPredictor):
            pass
        self.assertNotEqual(original, optimizer._generate_structure_id(
            self.candidate(compiled(_OtherPredictor()))))

    def test_all_fixed_or_singleton_preserves_bayesian_candidate(self):
        for fixed in (True, False):
            with self.subTest(fixed=fixed):
                step = compiled(_SearchPredictor(), fixed=('depth', 'locked') if fixed else ('locked',))
                if not fixed:
                    step.configuration['depth']['range'] = [3, 3]
                    step._flow_parameters['depth']['domain'] = [3, 3]
                candidate = self.candidate(step)
                optimizer = BayesianOptimizer()
                self.assertEqual(optimizer.run([candidate]), [candidate])
                self.assertIs(optimizer.run([candidate])[0], candidate)
                self.assertEqual(optimizer.skopt_optimizers, {})

    def test_bayesian_signature_normalizes_numpy_scalar_types(self):
        optimizer = BayesianOptimizer()
        candidate = self.candidate()
        original = optimizer._generate_structure_id(candidate)
        candidate.pipeline.predictor[1].configure('depth', np.int64(8))
        self.assertEqual(original, optimizer._generate_structure_id(candidate))

    def test_bayesian_local_variants_have_separate_stable_spaces(self):
        first = compiled(_SearchPredictor(), variant='first', alias='small')
        second = compiled(_SearchPredictor(), variant='second', alias='large')
        first._flow_variant_key, second._flow_variant_key = 0, 1
        alternatives = (deepcopy(first), deepcopy(second))
        first._flow_alternatives = second._flow_alternatives = alternatives
        first.is_interchangeable = second.is_interchangeable = True
        optimizer = BayesianOptimizer()
        one, two = self.candidate(first), self.candidate(second)
        original = optimizer._generate_structure_id(one)
        self.assertNotEqual(original, optimizer._generate_structure_id(two))
        self.assertNotEqual(one.pipeline.fingerprint(), two.pipeline.fingerprint())
        optimizer._initialize_search_space([one, two])
        self.assertEqual(len(optimizer.skopt_optimizers), 2)

        first.configure('depth', 8)
        alternatives[0].configure('depth', 5)
        alternatives[1].configure('depth', 9)
        first._flow_alias = 'renamed'
        first._flow_variant_id = 'fresh-id'
        self.assertEqual(original, optimizer._generate_structure_id(one))

    def test_random_and_bayesian_include_resamplers(self):
        candidate = self.candidate(resampler=compiled(_SearchResampler()))
        with patch('iaml.optimizers.random_optimizer.random.uniform', return_value=7):
            result = RandomOptimizer()._randomize_hyperparameters(deepcopy(candidate))
        self.assertEqual(result.pipeline.resamplers[0][1].get_config('depth'), 7)
        self.assertEqual(result.pipeline.resamplers[0][1].get_config('locked'), 4)
        optimizer = BayesianOptimizer()
        optimizer._initialize_search_space([candidate])
        backend = next(iter(optimizer.skopt_optimizers.values()))
        self.assertEqual(backend['param_keys'], ['0_depth', '1_depth'])

    def test_genetic_uses_only_local_configured_variants(self):
        first = compiled(_SearchTransform(), variant='first')
        second = compiled(_SearchTransform(), variant='second', alias='robust')
        second.configure('locked', 9)
        second.configuration['depth']['range'] = [1, 4]
        second._flow_parameters['depth']['domain'] = [1, 4]
        first._flow_alternatives = (deepcopy(first), deepcopy(second))
        first.is_interchangeable = True
        with patch.object(first, 'step_with_same_tags', side_effect=AssertionError('global registry')):
            changed = GeneticOptimizer._GeneticOptimizer__interchange(first)
        self.assertIs(type(changed), _SearchTransform)
        self.assertEqual(changed._flow_variant_id, 'second')
        self.assertEqual(changed._flow_alias, 'robust')
        self.assertEqual(changed.get_config('locked'), 9)
        self.assertEqual(changed.configuration['depth']['range'], [1, 4])
        changed.configure('locked', 6)
        self.assertEqual(second.get_config('locked'), 9)

    def test_optional_void_is_available_only_when_declared(self):
        step = compiled(_SearchTransform())
        step._flow_alternatives = (deepcopy(step),)
        step.is_interchangeable = True
        self.assertIs(GeneticOptimizer._GeneticOptimizer__interchange(step), step)
        void = VoidStep(step_to_mimic=deepcopy(step))
        void._flow_explicit = True
        void._flow_variant_id = 'absent'
        void._flow_parameters = {}
        step._flow_alternatives = (deepcopy(step), void)
        changed = GeneticOptimizer._GeneticOptimizer__interchange(step)
        self.assertIsInstance(changed, VoidStep)
        self.assertEqual(changed._flow_choice_id, 'slot')

    def test_cache_fingerprint_tracks_policy_without_alias_or_ids(self):
        first = compiled(_SearchPredictor())
        second = deepcopy(first)
        second._flow_alias = 'another'
        second._flow_variant_id = 'another-id'
        self.assertEqual(first.fingerprint(), second.fingerprint())
        second._flow_parameters['depth']['fixed'] = True
        self.assertNotEqual(first.fingerprint(), second.fingerprint())

        pipeline = self.candidate(first).pipeline
        before = pipeline.fingerprint()
        first._flow_parameters['depth']['fixed'] = True
        self.assertNotEqual(before, pipeline.fingerprint())

    def test_cache_fingerprint_distinguishes_required_choices_and_explicit_domains(self):
        step = compiled(_SearchPredictor())
        before = step.fingerprint()
        step._flow_required = True
        self.assertNotEqual(step.fingerprint(), before)
        before = step.fingerprint()
        step._flow_parameters['depth']['explicit_domain'] = True
        self.assertNotEqual(step.fingerprint(), before)

        step._flow_alternatives = (deepcopy(step),)
        before = step.fingerprint()
        step._flow_alternatives[0]._flow_required = False
        self.assertNotEqual(step.fingerprint(), before)


class TestFlowFittedCache(unittest.TestCase):
    def setUp(self):
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        backend = CacheService()
        backend.__set_backend__({}, [], SimpleNamespace(value=False), RLock(), 100)
        cache.configure(backend)
        _SearchTransform.fit_calls = 0

    def test_fit_cache_reuses_fit_but_restores_current_recipe_provenance(self):
        X = pd.DataFrame({'x': [1., 2., 3., 4.]})
        y = np.array([0, 0, 0, 1])
        first = compiled(_SearchTransform(), variant='old-id', alias='old')
        first_pipe = IAMLPipeline([('transform', first)], estimator_type='classifier')
        first_X, _ = first_pipe.fit_transform(X.copy(), y.copy())
        second = compiled(_SearchTransform(), variant='new-id', alias='current')
        second_pipe = IAMLPipeline([('transform', second)], estimator_type='classifier')
        actual_X, _ = second_pipe.fit_transform(X.copy(), y.copy())
        pd.testing.assert_frame_equal(actual_X, first_X)
        restored = second_pipe.transformers[0][1]
        self.assertEqual(_SearchTransform.fit_calls, 1)
        self.assertEqual(restored._flow_alias, 'current')
        self.assertEqual(restored._flow_node_id, 'new-id')
        self.assertEqual(restored._flow_variant_id, 'new-id')
        self.assertEqual(first_pipe.transformers[0][1]._flow_alias, 'old')

    def test_inapplicable_required_variant_cannot_become_an_undeclared_void(self):
        step = compiled(_SearchTransform(), alias='required')
        step.is_interchangeable = True
        step._flow_alternatives = (deepcopy(step),)
        pipeline = IAMLPipeline([('transform', step)], estimator_type='classifier')
        with patch.object(step, 'suitable', return_value=False):
            with self.assertRaisesRegex(ValueError, 'required.*inapplicable'):
                pipeline.fit_transform(pd.DataFrame({'x': [1., 2.]}), [0, 1])

    def test_inapplicable_optional_variant_uses_local_void(self):
        step = compiled(_SearchTransform(), alias='optional')
        void = compiled(VoidStep(step_to_mimic=deepcopy(step)), variant='absent')
        step.is_interchangeable = True
        step._flow_alternatives = (deepcopy(step), void)
        pipeline = IAMLPipeline([('transform', step)], estimator_type='classifier')
        X = pd.DataFrame({'x': [1., 2.]})
        with patch.object(step, 'suitable', return_value=False):
            actual_X, _ = pipeline.fit_transform(X.copy(), [0, 1])
        pd.testing.assert_frame_equal(actual_X, X)
        restored = pipeline.transformers[0][1]
        self.assertIsInstance(restored, VoidStep)
        self.assertEqual(restored._flow_variant_id, 'absent')
        self.assertEqual(restored._flow_choice_id, 'slot')

    def test_inapplicable_standalone_component_keeps_historical_skip(self):
        step = compiled(_SearchTransform(), alias='adapted')
        pipeline = IAMLPipeline([('transform', step)], estimator_type='classifier')
        X = pd.DataFrame({'x': [1., 2.]})
        with patch.object(step, 'suitable', return_value=False):
            actual_X, _ = pipeline.fit_transform(X.copy(), [0, 1])
        pd.testing.assert_frame_equal(actual_X, X)
        self.assertEqual(pipeline.training_steps, [])


if __name__ == '__main__':
    unittest.main()
