"""Train and inspect complete studies through the declarative public API."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from iaml import AccuracyMetric, IAML, MeanStatistic, RecallMetric
from iaml.cache import Cache
from iaml.candidate import Candidate
from iaml.dataset import Dataset
from iaml.explainers import KernelSHAP
from iaml.flow import Const, PipelineSpec, choice, explanations, metrics, statistics, use
from iaml.logger import Logger
from iaml.optimizers import Optimizer
from iaml.sklearn_preprocessor import SklearnPreprocessor
from iaml.steps import DecisionTreeClassifier


class NoSearchOptimizer(Optimizer):
    def __init__(self, duration=None):
        super().__init__()


class FlowStudyIntegrationTests(unittest.TestCase):
    def setUp(self):
        logger = Logger()
        self.addCleanup(setattr, logger, 'verbose', logger.verbose)
        logger.verbose = 0
        Cache().configure(None)
        self.X = pd.DataFrame({'value': np.arange(24, dtype=float),
                               'other': np.tile([0.0, 1.0], 12)})
        self.y = pd.Series(np.tile([0, 1], 12))

    def make_study(self, **kwargs):
        options = dict(
            pipeline=use(DecisionTreeClassifier, max_depth=Const(2)).named('tree'),
            metrics=metrics(use(AccuracyMetric).named('quality')),
            main_metric='quality', statistics=[], explanations=[],
            optimizer=NoSearchOptimizer, max_workers=1,
            max_duration=30, max_stage_duration=10,
        )
        options.update(kwargs)
        return IAML(**options)

    def test_explicit_pipeline_trains_without_implicit_candidates(self):
        study = self.make_study()
        model = study.fit(self.X, self.y, verbose=0)[0]
        self.assertEqual(len(study.candidates), 1)
        self.assertEqual(set(model.computed_metrics), {'quality'})
        self.assertEqual(model.pipeline.predictor[1].get_config('max_depth'), 2)
        scores = deepcopy(model.computed_metrics)
        self.assertEqual(set(model.evaluate(self.X, self.y)), {'quality'})
        self.assertEqual(model.computed_metrics, scores)
        self.assertEqual(model.predict(self.X).shape, (24,))
        report = study.describe()
        self.assertEqual(report['status'], 'resolved')
        self.assertIn('evaluated_candidates', report)
        self.assertIn('versions', report)
        self.assertTrue(report['metrics']['tree']['resolved'])
        self.assertTrue(report['statistics']['tree']['resolved'])
        self.assertEqual(report['compiled_analyses']['metrics'][0]['key'], 'quality')





    def test_analytic_and_recipe_snapshots_survive_later_edits(self):
        recipe = use(DecisionTreeClassifier, max_depth=Const(2)).named('tree')
        original_metrics = metrics(use(RecallMetric, pos_label=1).named('recall'))
        study = self.make_study(
            pipeline=recipe, metrics=original_metrics, main_metric='recall',
            explanations=explanations(use(KernelSHAP, nsamples=4).named('shap')),
        )
        recipe.configure(max_depth=Const(3))
        original_metrics['recall'].configure(pos_label=0)
        self.assertEqual(study.pipeline.parameters['max_depth'].value, 2)
        model = study.fit(self.X, self.y, verbose=0)[0]
        study.metrics['recall'].configure(pos_label=0)
        study.explanations['shap'].configure(nsamples=8)
        study.pipeline.configure(max_depth=Const(4))
        self.assertEqual(model.metrics[0].pos_label, 1)
        self.assertEqual(model.pipeline.predictor[1].get_config('max_depth'), 2)
        with patch.object(model.pipeline, 'explain_model', return_value='explained') as explain:
            self.assertEqual(model.explain(self.X.iloc[:2]), {'shap': 'explained'})
            self.assertEqual(explain.call_args.kwargs['nsamples'], 4)

    def test_default_recipe_exposes_minimal_branch_and_removal_is_effective(self):
        study = IAML(max_workers=1, max_stage_duration=1)
        self.assertEqual([node.alias for node in study.pipeline], ['minimal', 'main'])
        recipe = PipelineSpec.default()
        recipe.remove('minimal')
        configured = self.make_study(pipeline=recipe)
        self.assertEqual([node.alias for node in configured.pipeline], ['main'])
        with self.assertRaises(KeyError):
            configured.pipeline['minimal']
        self.assertIsNone(configured.minimal_predictor_step)

    def test_generated_branches_follow_the_visible_recipe_and_imputation(self):
        recipe = PipelineSpec.default()
        recipe.main.replace(use(DecisionTreeClassifier, max_depth=Const(2)))
        recipe.minimal_predictor.replace(use(DecisionTreeClassifier, max_depth=Const(3)))
        study = self.make_study(pipeline=recipe)
        missing = self.X.copy()
        missing.iloc[0, 0] = np.nan
        source = Candidate(Dataset(missing, self.y))
        outputs = study._flow_execution_root.run(source)
        self.assertEqual([c.pipeline.predictor[1].get_config('max_depth') for c in outputs],
                         [3, 2])
        self.assertEqual(outputs[0].pipeline.predictor[1]._flow_alias, 'minimal_predictor')
        self.assertFalse(outputs[0].dataset.X.isna().values.any())
        study.pipeline.remove('minimal')
        study._prepare_flow()
        remaining = study._flow_execution_root.run(source)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].pipeline.predictor[1].get_config('max_depth'), 2)

    def test_descriptive_requests_preserve_search_dataset_and_configuration(self):
        study = self.make_study(statistics=statistics(use(MeanStatistic).named('mean')))
        before = study.get_descriptive_statistics(self.X, self.y)
        self.assertIsNone(study._last_dataset)
        self.assertTrue(study.get_descriptive_statistics().empty)
        study.fit(self.X, self.y, verbose=0)
        raw = study.get_descriptive_statistics()
        changed = self.X + 100
        different = study.get_descriptive_statistics(changed, self.y)
        self.assertFalse(before.equals(different))
        pd.testing.assert_frame_equal(raw, study.get_descriptive_statistics())
        raw.iloc[0, 0] = -999
        self.assertNotEqual(study.get_descriptive_statistics().iloc[0, 0], -999)

    def test_root_collection_replace_returns_attached_object(self):
        study = self.make_study()
        old = study.metrics
        replacement = old.replace(metrics(use(RecallMetric).named('recall')))
        self.assertIs(study.metrics, replacement)
        self.assertIsNone(old._owner)

    def test_historical_configuration_is_preserved_when_recipe_is_compiled(self):
        study = self.make_study()
        step_id = study.all_configurations()[0]['step_id']
        study.configure_all({step_id: {'max_depth': 3}})
        self.assertEqual(study.pipeline.parameters['max_depth'].value, 3)
        model = study.fit(self.X, self.y, verbose=0)[0]
        self.assertEqual(model.pipeline.predictor[1].get_config('max_depth'), 3)

    def test_explicit_analytics_are_respected_with_a_legacy_execution_tree(self):
        study = self.make_study(main_metric='recall')
        study.metrics = metrics(use(RecallMetric).named('recall'))
        study.first_step = DecisionTreeClassifier()
        model = study.fit(self.X, self.y, verbose=0)[0]
        self.assertEqual(set(model.computed_metrics), {'recall'})
        self.assertEqual(model.explain(self.X.iloc[:2]), {})

    def test_analysis_collection_cannot_be_assigned_as_a_pipeline(self):
        study = self.make_study()
        original = study.pipeline
        with self.assertRaises(TypeError):
            study.pipeline = metrics()
        self.assertIs(study.pipeline, original)

    def test_incomplete_recipe_can_be_attached_and_completed_before_training(self):
        study = self.make_study(pipeline=choice())
        self.assertEqual(list(study.pipeline), [])
        with self.assertRaisesRegex(ValueError, 'has no alternatives'):
            study.fit(self.X, self.y, verbose=0)
        self.assertIsNone(study.executor)
        study.pipeline.add(use(DecisionTreeClassifier, max_depth=Const(2)).named('tree'))
        model = study.fit(self.X, self.y, verbose=0)[0]
        self.assertEqual(len(study.candidates), 1)
        self.assertEqual(model.pipeline.predictor[1].get_config('max_depth'), 2)

    def test_default_metric_family_edits_are_respected_with_legacy_execution(self):
        study = self.make_study(metrics=None, main_metric='recall')
        next(iter(study.metrics.find_all(RecallMetric))).named('recall').configure(pos_label=0)
        study.first_step = DecisionTreeClassifier()
        model = study.fit(self.X, self.y, verbose=0)[0]
        self.assertIn('recall', model.computed_metrics)
        configured = next(metric for metric in model.metrics if type(metric) is RecallMetric)
        self.assertEqual(configured.pos_label, 0)

    def test_constructor_rejects_non_recipe_pipeline(self):
        with self.assertRaisesRegex(TypeError, 'training recipe'):
            self.make_study(pipeline='invalid')

    def test_absent_class_removal_and_empty_start_reset(self):
        from iaml.flow import choice
        from iaml.steps import RandomForestClassifier
        models = choice(use(DecisionTreeClassifier).named('tree'),
                        use(RandomForestClassifier).named('forest'))
        with self.assertRaises(ValueError):
            models.remove(type('Absent', (), {}))
        models.start('forest')
        with self.assertRaisesRegex(ValueError, r'start\(\)'):
            models.remove('forest')
        models.start().remove('forest')
        self.assertEqual([model.alias for model in models], ['tree'])


if __name__ == '__main__':
    unittest.main()
