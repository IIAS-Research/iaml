"""Check public availability and isolation of experimental components."""
import importlib
import unittest

import pandas as pd

import iaml
from iaml.decorators.all import find_steps_by_tag
from iaml.iaml import IAML
from iaml.metastep import MetaStep


class TestComponentRegistry(unittest.TestCase):
    """Explicit prototype imports must not alter the default search space."""

    def test_hist_gradient_boosting_regressor_is_discovered(self):
        step_class = iaml.ActHistGradientBoostingRegressor
        self.assertIn(step_class, find_steps_by_tag('predictor'))
        stage = MetaStep(tag='regressor')
        self.assertTrue(any(isinstance(step, step_class) for step in stage.steps))

    def test_experimental_steps_are_not_public_or_automatically_discovered(self):
        components = (
            ('cleaning.act_encode_target_column', 'ActCategoryStringToNumeric'),
            ('features_precleaning.act_drop_bad_quality_rows', 'ActDropBadQualityRows'),
            ('features_preprocessing.act_polynomial_features', 'ActPolynomialFeatures'),
            ('features_preprocessing.act_cyclical_date_encoding', 'ActCyclicalDateEncoding'),
        )
        automatic_tags = ('cleaning', 'features_precleaning', 'features_preprocessing',
                          'normalize', 'predictor')
        for module_path, class_name in components:
            with self.subTest(component=class_name):
                module = importlib.import_module(f'iaml.actionables.{module_path}')
                step_class = getattr(module, class_name)
                self.assertFalse(hasattr(iaml, class_name))
                package = importlib.import_module(module.__package__)
                self.assertFalse(hasattr(package, class_name))
                self.assertEqual(step_class().tags, {'experimental'})
                for tag in automatic_tags:
                    self.assertNotIn(step_class, find_steps_by_tag(tag))

    def test_importing_experimental_auc_does_not_activate_it(self):
        module = importlib.import_module('iaml.metrics.cumulative_dynamic_auc')
        metric_class = module.CumulativeDynamicAUCMetric
        self.assertFalse(hasattr(iaml, 'CumulativeDynamicAUCMetric'))
        X = pd.DataFrame({'feature': [0.1, 0.2, 0.3]})
        y = [(True, 1.0), (False, 2.0), (True, 3.0)]
        automl = IAML(max_workers=1, max_stage_duration=1)
        metrics = automl._IAML__metrics_selection(X, y, 'survival')
        self.assertFalse(any(isinstance(metric, metric_class) for metric in metrics))


if __name__ == '__main__':
    unittest.main()
