"""Storage and prediction contracts for both survival forest adapters."""
import unittest

import numpy as np
from sksurv.metrics import integrated_brier_score

from iaml.actionables.predictors.survival.act_extra_survival_trees import ActExtraSurvivalTrees
from iaml.actionables.predictors.survival.act_random_survival_forest import ActRandomSurvivalForest
from iaml.dataset import Dataset
from iaml.flow import Const, use
from tests.helpers.datasets import make_survival_data


FORESTS = (ActExtraSurvivalTrees, ActRandomSurvivalForest)


class TestSurvivalForestMemory(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(*make_survival_data(n_samples=96, seed=43))

    def fit_forest(self, forest_class, low_memory="auto", requirements=None, **parameters):
        step = forest_class()
        step.configure({"n_estimators": 5, "random_state": 41, "low_memory": low_memory,
                        **parameters})
        step.configure_prediction_requirements(requirements)
        self.assertIs(step.fit(self.dataset), step)
        return step

    def test_low_memory_preserves_risk_scores_and_tree_structure(self):
        """Scalar storage must change neither split decisions nor risk scores."""
        for forest_class in FORESTS:
            with self.subTest(forest=forest_class.__name__):
                full = self.fit_forest(forest_class, low_memory=False)
                compact = self.fit_forest(forest_class, low_memory=True)
                np.testing.assert_allclose(compact.predict(self.dataset.X),
                                           full.predict(self.dataset.X), rtol=1e-12, atol=1e-12)
                for full_tree, compact_tree in zip(full.model.estimators_,
                                                   compact.model.estimators_):
                    np.testing.assert_array_equal(full_tree.tree_.children_left,
                                                  compact_tree.tree_.children_left)
                    np.testing.assert_array_equal(full_tree.tree_.feature,
                                                  compact_tree.tree_.feature)
                    np.testing.assert_array_equal(full_tree.tree_.threshold,
                                                  compact_tree.tree_.threshold)
                    self.assertEqual(compact_tree.tree_.value.shape[1:], (1, 1))
                    self.assertGreater(full_tree.tree_.value.nbytes,
                                       compact_tree.tree_.value.nbytes)

    def test_auto_enables_scalar_storage_only_for_known_risk_only_contract(self):
        for forest_class in FORESTS:
            for requirements, expected in (({"predict"}, True), (None, False), (set(), False),
                                           ({"predict", "unknown_custom_output"}, False)):
                with self.subTest(forest=forest_class.__name__, requirements=requirements):
                    step = self.fit_forest(forest_class, requirements=requirements)
                    self.assertIs(step.model.low_memory, expected)
                    self.assertEqual(step.get_config("low_memory"), "auto")

    def test_curve_requirements_preserve_survival_hazard_and_ibs(self):
        """Any requested curve keeps both functions and Brier-based scoring available."""
        for forest_class in FORESTS:
            for method in ("predict_survival_function", "predict_cumulative_hazard_function"):
                with self.subTest(forest=forest_class.__name__, method=method):
                    step = self.fit_forest(forest_class, requirements={"predict", method})
                    self.assertIs(step.model.low_memory, False)
                    X, y = self.dataset.to_survival()
                    time_field = y.dtype.names[1]
                    times = np.quantile(y[time_field], [0.2, 0.4, 0.6])
                    survival = step.predict_survival_function(X)
                    hazards = step.predict_cumulative_hazard_function(X)
                    probabilities = np.asarray([curve(times) for curve in survival])
                    self.assertTrue(np.isfinite(probabilities).all())
                    self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
                    self.assertTrue(np.all(np.diff(probabilities, axis=1) <= 0))
                    self.assertTrue(np.isfinite([curve(times) for curve in hazards]).all())
                    self.assertTrue(np.isfinite(integrated_brier_score(y, y, probabilities, times)))

    def test_explicit_full_storage_keeps_curves_with_risk_only_contract(self):
        for forest_class in FORESTS:
            with self.subTest(forest=forest_class.__name__):
                step = self.fit_forest(forest_class, low_memory=False, requirements={"predict"})
                self.assertIs(step.model.low_memory, False)
                self.assertEqual(len(step.predict_survival_function(self.dataset.X)), 96)

    def test_explicit_low_memory_rejects_required_curves_before_fitting(self):
        for forest_class in FORESTS:
            for method in ("predict_survival_function", "predict_cumulative_hazard_function"):
                with self.subTest(forest=forest_class.__name__, method=method):
                    step = forest_class()
                    step.configure({"low_memory": True})
                    step.configure_prediction_requirements({"predict", method})
                    with self.assertRaisesRegex(ValueError, method):
                        step.fit(self.dataset)
                    self.assertIsNone(step.model)

    def test_risk_only_fit_explains_how_to_enable_later_curve_predictions(self):
        for forest_class in FORESTS:
            with self.subTest(forest=forest_class.__name__):
                step = self.fit_forest(forest_class, requirements={"predict"})
                for method in ("predict_survival_function", "predict_cumulative_hazard_function"):
                    with self.assertRaisesRegex(ValueError, "low_memory=False.*refit"):
                        getattr(step, method)(self.dataset.X)

    def test_automatic_policy_is_resolved_again_for_a_different_fit(self):
        for forest_class in FORESTS:
            with self.subTest(forest=forest_class.__name__):
                step = self.fit_forest(forest_class, requirements={"predict"})
                step.configure_prediction_requirements({"predict_survival_function"})
                step.fit(self.dataset)
                self.assertIs(step.model.low_memory, False)
                self.assertEqual(len(step.predict_survival_function(self.dataset.X)), 96)

    def test_tree_limits_are_bounded_by_default_and_configurable(self):
        for forest_class in FORESTS:
            with self.subTest(forest=forest_class.__name__):
                step = forest_class()
                self.assertEqual(step.get_config("max_depth"), 12)
                self.assertEqual(step.get_config("max_leaf_nodes"), 64)
                self.assertEqual(step.get_config("min_samples_leaf"), 5)
                self.assertEqual(step.get_config("min_samples_split"), 10)
                step = self.fit_forest(forest_class, max_depth=3, max_leaf_nodes=4,
                                       min_samples_leaf=2, min_samples_split=4)
                for tree in step.model.estimators_:
                    self.assertLessEqual(tree.tree_.max_depth, 3)
                    self.assertLessEqual(tree.tree_.n_leaves, 4)
                unbounded = self.fit_forest(forest_class, low_memory=True,
                                           max_depth=None, max_leaf_nodes=None)
                self.assertIsNone(unbounded.model.max_depth)
                self.assertIsNone(unbounded.model.max_leaf_nodes)

    def test_recipes_accept_automatic_and_explicit_modes(self):
        for forest_class in FORESTS:
            for low_memory in ("auto", True, False):
                with self.subTest(forest=forest_class.__name__, low_memory=low_memory):
                    recipe = use(forest_class, low_memory=low_memory,
                                 max_depth=Const(None), max_leaf_nodes=Const(None))
                    self.assertEqual(recipe.parameters["low_memory"].value, low_memory)
                    self.assertIsNone(recipe.parameters["max_depth"].value)
                    self.assertIsNone(recipe.parameters["max_leaf_nodes"].value)
            with self.subTest(forest=forest_class.__name__, low_memory="invalid"):
                with self.assertRaises(ValueError):
                    use(forest_class, low_memory="invalid")



if __name__ == "__main__":
    unittest.main()
