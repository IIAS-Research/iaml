"""Generic CatBoost domains and conditional backend parameters."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from iaml.actionables.predictors.classifier.act_catboost_classifier import ActCatBoost
from iaml.actionables.predictors.regressor.act_catboost_regressor import ActCatBoostRegressor
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.iaml_pipeline import IAMLPipeline
from iaml.optimizers import BayesianOptimizer, GeneticOptimizer, RandomOptimizer


class CatBoostSearchSpaceTests(unittest.TestCase):
    """Check both tasks without tuning against model quality or a private dataset."""

    step_types = (ActCatBoost, ActCatBoostRegressor)
    policies = ('SymmetricTree', 'Depthwise', 'Lossguide')
    secondary = ('eval_metric', 'border_count', 'bootstrap_type', 'leaf_estimation_iterations')

    def test_common_domains_and_default_types(self):
        expected = {
            'iterations': ([300, 10000], 1000, int),
            'learning_rate': ([0.005, 0.20], 0.03, float),
            'depth': ([3, 10], 6, int),
            'l2_leaf_reg': ([1e-3, 1e3], 3.0, float),
            'random_strength': ([0.0, 5.0], 1.0, float),
            'min_data_in_leaf': ([1, 300], 1, int),
        }
        for step_type in self.step_types:
            step = step_type()
            with self.subTest(step=step_type.__name__):
                for key, (bounds, default, value_type) in expected.items():
                    config = step.configuration[key]
                    self.assertEqual(config['range'], bounds)
                    self.assertEqual(step.get_config(key), default)
                    self.assertIs(type(step.get_config(key)), value_type)
                self.assertEqual(step.configuration['grow_policy']['categorical'],
                                 list(self.policies))
                self.assertEqual(step.get_config('grow_policy'), 'SymmetricTree')
                self.assertEqual(step.priorize(), 0.5)

    def test_leaf_size_is_sent_only_to_compatible_backend_policies(self):
        for step_type in self.step_types:
            step = step_type()
            step.configure({'min_data_in_leaf': 7})
            for policy in self.policies:
                with self.subTest(step=step_type.__name__, policy=policy):
                    step.configure({'grow_policy': policy})
                    active = policy != 'SymmetricTree'
                    params = step.passthrough_parameters()
                    self.assertEqual('min_data_in_leaf' in params, active)
                    if active:
                        self.assertEqual(params['min_data_in_leaf'], 7)

    def test_secondary_parameters_use_backend_defaults_and_keep_explicit_values(self):
        configured = dict(zip(self.secondary, ('Accuracy', 32, 'Bernoulli', 2)))
        for step_type in self.step_types:
            with self.subTest(step=step_type.__name__):
                step = step_type()
                for key in self.secondary:
                    self.assertIsNone(step.get_config(key))
                    self.assertNotIn(key, step.passthrough_parameters())
                step.configure(configured)
                self.assertEqual({key: step.passthrough_parameters()[key]
                                  for key in self.secondary}, configured)

    def test_regression_search_keeps_rmse_and_preserves_explicit_loss(self):
        step = ActCatBoostRegressor()
        self.assertEqual(step.passthrough_parameters()['loss_function'], 'RMSE')
        self.assertNotIn('range', step.configuration['loss_function'])
        self.assertNotIn('categorical', step.configuration['loss_function'])
        step.configure({'loss_function': 'MAE'})
        self.assertEqual(step.passthrough_parameters()['loss_function'], 'MAE')

    def test_existing_optimizers_keep_numeric_proposals_in_range_and_typed(self):
        features = pd.DataFrame({'feature': np.arange(12, dtype=float)})
        for step_type in self.step_types:
            for optimizer_type in (RandomOptimizer, GeneticOptimizer, BayesianOptimizer):
                with self.subTest(step=step_type.__name__, optimizer=optimizer_type.__name__):
                    step = step_type()
                    step.optimizable = True
                    regression = step_type is ActCatBoostRegressor
                    dataset = Dataset(features, np.arange(12) * 0.17 if regression
                                      else np.arange(12) % 2)
                    pipeline = IAMLPipeline([('catboost', step)],
                                            estimator_type='regressor' if regression
                                            else 'classifier')
                    candidate = Candidate(dataset, iaml_pipeline=pipeline)
                    candidate.computed_metrics = {candidate.main_metric: 0.5}
                    optimizer = (optimizer_type(nb_candidate=4)
                                 if optimizer_type is GeneticOptimizer else optimizer_type())
                    proposals = optimizer.run([candidate])
                    self.assertTrue(proposals)
                    for proposal in proposals:
                        model = proposal.pipeline.predictor[1]
                        for key, config in model.configuration.items():
                            if 'range' in config:
                                value = model.get_config(key)
                                self.assertGreaterEqual(value, config['range'][0])
                                self.assertLessEqual(value, config['range'][1])
                                self.assertIs(type(value), type(config['default']))
                        for key in self.secondary:
                            self.assertIsNone(model.get_config(key))
                        if regression:
                            self.assertEqual(model.get_config('loss_function'), 'RMSE')

    def test_explicit_compatible_classification_losses_are_preserved(self):
        features = pd.DataFrame({'feature': np.arange(12, dtype=float)})
        for class_count, loss in ((2, 'CrossEntropy'), (3, 'MultiClassOneVsAll')):
            with self.subTest(loss=loss):
                step = ActCatBoost()
                step.configure({'loss_function': loss, 'iterations': 2, 'depth': 2})
                with patch(ActCatBoost.__module__ + '.CatBoostClassifier') as constructor:
                    step.fit(Dataset(features, np.arange(12) % class_count))
                self.assertEqual(constructor.call_args.kwargs['loss_function'], loss)
                self.assertEqual(constructor.call_args.kwargs['iterations'], 2)
                self.assertEqual(constructor.call_args.kwargs['depth'], 2)

    def test_default_losses_and_all_growing_policies_fit_on_cpu(self):
        features = pd.DataFrame(np.arange(240, dtype=float).reshape(60, 4))
        cases = ((ActCatBoostRegressor, CatBoostRegressor, np.arange(60) * 0.17, 'RMSE'),
                 (ActCatBoost, CatBoostClassifier, np.arange(60) % 2, 'Logloss'),
                 (ActCatBoost, CatBoostClassifier, np.arange(60) % 3, 'MultiClass'))
        for step_type, constructor, target, loss in cases:
            for policy in self.policies:
                with self.subTest(step=step_type.__name__, loss=loss, policy=policy):
                    step = step_type()
                    # Explicit backend-valid values remain legal outside the search domain.
                    step.configure({'iterations': 2, 'depth': 2, 'grow_policy': policy,
                                    'min_data_in_leaf': 3})
                    backend = step_type.__module__ + (
                        '.CatBoostRegressor' if step_type is ActCatBoostRegressor
                        else '.CatBoostClassifier')

                    def build_model(**params):
                        return constructor(thread_count=1, allow_writing_files=False, **params)

                    with patch(backend, side_effect=build_model):
                        step.fit(Dataset(features, target))
                    params = step.model.get_params()
                    self.assertEqual(params['loss_function'], loss)
                    self.assertEqual(params['iterations'], 2)
                    self.assertEqual(params['depth'], 2)
                    self.assertEqual(params['grow_policy'], policy)
                    self.assertEqual('min_data_in_leaf' in params, policy != 'SymmetricTree')
                    self.assertTrue(np.isfinite(step.predict(features)).all())
                    for key in self.secondary:
                        self.assertNotIn(key, params)


if __name__ == '__main__':
    unittest.main()
