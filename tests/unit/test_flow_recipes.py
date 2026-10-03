"""Behavioral contracts for declarative recipe editing and reconstruction."""
import pickle
import unittest
from types import SimpleNamespace
from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval

from iaml import Step
from iaml.flow import Const, Float, Int, choice, use
from iaml.steps import (DecisionTreeClassifier, RandomForestClassifier, StandardScaler,
                        RobustScaler, MinMaxScaler, SMOTE, SMOTETomek, SMOTEENN,
                        SimpleImputer, Normalizer, UnitNormScaler)
from iaml.metrics import RecallMetric
from iaml.statistics import TopKValueCountsStatistic


class FlowRecipeTests(unittest.TestCase):
    def forest(self, alias, low, high):
        return use(RandomForestClassifier, n_estimators=Int(low, high, initial=low)).named(alias)


    def test_constants_retain_domain_and_literal_reactivates(self):
        forest = self.forest("forest", 100, 500)
        forest.configure(n_estimators=Const(700))
        fixed = forest.instantiate()
        self.assertEqual(fixed.get_config("n_estimators"), 700)
        self.assertNotIn("range", fixed.configuration["n_estimators"])
        self.assertEqual(fixed._flow_parameters["n_estimators"]["domain"], [100, 500])
        forest.configure(n_estimators=350)
        self.assertFalse(forest.parameters["n_estimators"].fixed)
        self.assertEqual(forest.instantiate().configuration["n_estimators"]["range"], [100, 500])
        with self.assertRaises(ValueError):
            forest.configure(n_estimators=700)
        self.assertEqual(forest.parameters["n_estimators"].value, 350)






    def test_unit_norm_scaler_loads_current_and_historical_pipeline_names(self):
        step = use(UnitNormScaler, norm=Const("l1")).instantiate()
        step.enable = False
        payload = step.json_pipeline()
        self.assertEqual(payload["step"], "ActUnitNormScaler")

        for class_name in ("ActUnitNormScaler", "ActNormalizer"):
            with self.subTest(class_name=class_name):
                restored = Step.from_pipeline({**payload, "step": class_name})
                self.assertIs(type(restored), UnitNormScaler)
                self.assertEqual(restored.get_config("norm"), "l1")
                self.assertFalse(restored.enable)

    def test_unit_norm_scaler_loads_historical_pickles(self):
        step = use(UnitNormScaler, norm=Const("l1")).instantiate()
        payload = pickle.dumps(step, protocol=0)
        current = b"ciaml.actionables.normalize.act_normalizer\nActUnitNormScaler\n"
        historical = b"ciaml.actionables.normalize.act_normalizer\nActNormalizer\n"
        self.assertIn(current, payload)

        restored = pickle.loads(payload.replace(current, historical, 1))

        self.assertIs(type(restored), UnitNormScaler)
        self.assertEqual(restored.get_config("norm"), "l1")






    def test_domains_validate_types_and_initial_values(self):
        with self.assertRaises(ValueError):
            Int(2, 1, initial=1)
        with self.assertRaises(TypeError):
            Int(1, 3, initial=True)
        with self.assertRaises(ValueError):
            Float(0, 1, initial=float("nan"))
        with self.assertRaises(ValueError):
            use(RandomForestClassifier, n_estimators=Const(0))

    def test_reusing_an_unnamed_fragment_assigns_distinct_node_ids(self):
        original = use(StandardScaler)
        recipe = original >> original
        self.assertEqual(len({node.node_id for node in recipe}), 2)





    def test_custom_components_can_declare_constraints_or_keep_default_type_checks(self):
        class Custom(Step):
            _flow_parameter_constraints = {"size": [Interval(Real, 0, None, closed="left"), None]}

            def __init__(self):
                super().__init__()
                self.configuration = {"size": {"default": 1}, "count": {"default": 2}}

        recipe = use(Custom, size=Const(None))
        recipe.configure(size=Const(0.5))
        self.assertEqual(recipe.instantiate().get_config("size"), 0.5)
        with self.assertRaises(ValueError):
            recipe.configure(size=Const(-1))
        with self.assertRaises(TypeError):
            recipe.configure(count=Const(1.5))








if __name__ == "__main__":
    unittest.main()
