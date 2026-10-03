"""Permutation selection follows the configured search objective."""
from functools import partial
import pickle
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from iaml.actionables.features_selection.act_permutation_importance_selector import (
    ActPermutationImportanceSelector,
)
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.cache import Cache
from iaml.cache_keys import hash_evaluation_context
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.logger import Logger
from iaml.metrics import (BalancedAccuracyMetric, MeanAbsoluteErrorMetric,
                          PrecisionMetric, RocAucMetric)
from iaml.splitters import kfold_splitter
from iaml.study_analyses import CompiledAnalysis


class TestPermutationObjective(unittest.TestCase):
    def setUp(self):
        X, self.y = make_classification(n_samples=80, n_features=4, n_informative=2,
                                        n_redundant=0, weights=[0.8, 0.2],
                                        random_state=31)
        self.X = pd.DataFrame(X, columns=list('abcd'))
        cache = Cache()
        self.addCleanup(cache.configure, cache._backend)
        cache.configure(None)
        verbose = Logger().verbose
        self.addCleanup(setattr, Logger(), 'verbose', verbose)
        Logger().verbose = 0

    def make_step(self):
        step = ActPermutationImportanceSelector()
        step.configure({'threshold': None, 'max_features': 4, 'n_repeats': 3,
                        'tree_n_estimators': 10, 'tree_max_depth': 3, 'random_state': 5})
        self.addCleanup(step.reset_cache)
        return step

    def assert_importances_match(self, step, dataset, scoring):
        expected = permutation_importance(step.estimator, dataset.X, dataset.y,
                                          scoring=scoring, n_repeats=3,
                                          random_state=5, n_jobs=1)
        np.testing.assert_allclose(list(step.importances.values()), expected.importances_mean)

    def test_auto_matches_auc_and_balanced_accuracy(self):
        for metric, scoring in ((RocAucMetric(), 'roc_auc'),
                                (BalancedAccuracyMetric(), 'balanced_accuracy')):
            with self.subTest(objective=scoring):
                dataset = Dataset(self.X.copy(), self.y.copy())
                step = self.make_step()
                step.run(Candidate(dataset, metrics=[metric], main_metric=str(metric)))
                self.assert_importances_match(step, dataset, scoring)
                self.assertEqual(step.get_config('scoring'), 'auto')
                resolved = step.resume_configuration()['resolved_scoring']
                self.assertEqual(resolved['metric'], str(metric))

    def test_default_candidate_and_direct_fit_use_the_default_objective(self):
        for direct_fit in (False, True):
            with self.subTest(direct_fit=direct_fit):
                dataset = Dataset(self.X.copy(), self.y.copy())
                step = self.make_step()
                if direct_fit:
                    step.fit(dataset)
                else:
                    step.run(Candidate(dataset))
                self.assert_importances_match(step, dataset, 'balanced_accuracy')

    def test_explicit_scoring_takes_precedence(self):
        dataset = Dataset(self.X.copy(), self.y.copy())
        step = self.make_step()
        step.configure('scoring', 'balanced_accuracy')
        step.run(Candidate(dataset, main_metric=RocAucMetric()))
        self.assert_importances_match(step, dataset, 'balanced_accuracy')
        self.assertNotIn('resolved_scoring', step.resume_configuration())

    def test_compiled_alias_resolves_the_metric_definition(self):
        metric = RocAucMetric()
        definition = CompiledAnalysis('ranking', metric, {},
                                      hash_evaluation_context(type(metric), {}))
        dataset = Dataset(self.X.copy(), self.y.copy())
        step = self.make_step()
        step.run(Candidate(dataset, main_metric='ranking', metric_definitions=[definition]))
        self.assert_importances_match(step, dataset, 'roc_auc')
        self.assertEqual(step.resume_configuration()['resolved_scoring']['objective'], 'ranking')

    def test_metric_parameters_and_minimization_direction_are_preserved(self):
        dataset = Dataset(self.X.copy(), self.X['a'].to_numpy() * 2 + 0.3)
        step = self.make_step()
        step.run(Candidate(dataset, main_metric=MeanAbsoluteErrorMetric()))
        self.assert_importances_match(step, dataset, 'neg_mean_absolute_error')

        candidate = Candidate(Dataset(self.X.copy(), self.y.copy()),
                              main_metric=PrecisionMetric(pos_label=0))
        step = self.make_step()
        step.run(candidate)
        first = step.fingerprint()
        self.assertEqual(step.resume_configuration()['resolved_scoring']['configuration'],
                         {'pos_label': 0})
        candidate.main_metric = PrecisionMetric(pos_label=1)
        step.run(candidate)
        self.assertNotEqual(step.fingerprint(), first)

    def test_objective_changes_invalidate_cache_before_run(self):
        candidate = Candidate(Dataset(self.X.copy(), self.y.copy()),
                              main_metric=RocAucMetric())
        step = self.make_step()
        with patch.object(candidate, 'add_to_pipeline', return_value=candidate), \
                patch.object(step, 'fit', wraps=step.fit) as fit:
            step.run(candidate)
            auc_fingerprint = step.fingerprint()
            step.run(candidate)
            self.assertEqual(fit.call_count, 1)
            candidate.main_metric = BalancedAccuracyMetric()
            step.run(candidate)
            self.assertEqual(fit.call_count, 2)
            self.assertNotEqual(step.fingerprint(), auc_fingerprint)
            candidate.main_metric = RocAucMetric()
            step.run(candidate)
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(step.fingerprint(), auc_fingerprint)

    def test_objective_survives_pipeline_copies_pickle_and_cv_refits(self):
        dataset = Dataset(self.X.copy(), self.y.copy())
        step = self.make_step()
        candidate = step.run(Candidate(dataset, metrics=[RocAucMetric()],
                                       main_metric='ROC AUC'))[0]
        candidate.pipeline.set_model(ActDecisionTreeClassifier())
        copied = pickle.loads(candidate.pipeline.copy().pickle())
        copied_selector = copied.transformers[0][1]
        self.assertEqual(copied_selector.fingerprint(), step.fingerprint())
        scorer = copied_selector._resolve_scoring(dataset)
        self.assertAlmostEqual(scorer(step.estimator, self.X, self.y),
                               roc_auc_score(self.y, step.estimator.predict_proba(self.X)[:, 1]))

        module = 'iaml.actionables.features_selection.act_permutation_importance_selector'
        with patch(f'{module}.permutation_importance', wraps=permutation_importance) as importance:
            scores = candidate.training_evaluate(dataset,
                                                  splitter=partial(kfold_splitter, nb_folds=3),
                                                  cache_split=False)
        self.assertTrue(np.isfinite(scores['ROC AUC']))
        self.assertEqual(importance.call_count, 3)
        for call in importance.call_args_list:
            scorer = call.kwargs['scoring']
            self.assertIsInstance(scorer.metric, RocAucMetric)
            self.assertLess(len(scorer.y_train), len(self.y))

    def test_auto_is_resolved_independently_for_each_candidate_in_a_batch(self):
        candidates = [Candidate(Dataset(self.X.copy(), self.y.copy()), main_metric=metric)
                      for metric in (RocAucMetric(), BalancedAccuracyMetric())]
        step = self.make_step()
        outputs = step.run(candidates)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(step.candidate, outputs)
        first, second = (output.pipeline.transformers[0][1] for output in outputs)
        self.assertEqual(first.resume_configuration()['resolved_scoring']['metric'], 'ROC AUC')
        self.assertEqual(second.resume_configuration()['resolved_scoring']['metric'],
                         'balanced_accuracy')
        scorer = second._resolve_scoring(candidates[1].dataset)
        self.assertAlmostEqual(scorer(second.estimator, self.X, self.y),
                               balanced_accuracy_score(self.y, second.estimator.predict(self.X)))


if __name__ == '__main__':
    unittest.main()
