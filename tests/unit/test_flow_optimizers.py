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


    def test_explicit_domain_activates_only_its_parameter(self):
        step = compiled(_SearchPredictor())
        step.optimizable = False
        step._flow_parameters['locked']['fixed'] = False
        step._flow_parameters['locked']['optimizable'] = False
        self.assertEqual(parameter_keys(step), ['depth'])






    def test_const_does_not_clamp_to_inactive_domain(self):
        step = compiled(_SearchPredictor())
        step.configure('locked', 30)
        self.assertTrue(step.check_configuration())
        self.assertEqual(step.get_config('locked'), 30)









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
