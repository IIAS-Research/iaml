"""Regression tests for survival parameters reaching the fitted estimators."""
import unittest

import numpy as np

from iaml.actionables.predictors.survival.act_cox import ActCox
from iaml.actionables.predictors.survival.act_extra_survival_trees import ActExtraSurvivalTrees
from iaml.actionables.predictors.survival.act_gradient_boosting_survival_analysis import (
    ActGradientBoostingSurvivalAnalysis,
)
from iaml.actionables.predictors.survival.act_random_survival_forest import ActRandomSurvivalForest
from iaml.actionables.predictors.survival.act_survival_component_wise_gboost import (
    ActComponentwiseGradientBoostingSurvivalAnalysis,
)
from iaml.actionables.predictors.survival.act_survival_tree import ActSurvivalTree
from iaml.actionables.predictors.survival.act_survival_xgboost import (
    ActGradientBoostingSurvivalAnalysis as LegacyGradientBoostingSurvivalAnalysis,
    ActSurvivalXGBoost,
)
from iaml.dataset import Dataset
from tests.helpers.datasets import make_survival_data


ENSEMBLE_STEPS = (
    ActRandomSurvivalForest,
    ActExtraSurvivalTrees,
    ActGradientBoostingSurvivalAnalysis,
    ActComponentwiseGradientBoostingSurvivalAnalysis,
)
SURVIVAL_STEPS = ENSEMBLE_STEPS + (ActCox, ActSurvivalTree, ActSurvivalXGBoost)


class TestSurvivalHyperparameters(unittest.TestCase):
    """Check actual fitted models, including parameters with different backend defaults."""

    def setUp(self):
        X, y = make_survival_data(n_samples=48, seed=42)
        self.dataset = Dataset(X, y)

    def _fit_configured(self, step_class, parameters):
        step = step_class()
        step.configure(parameters)
        self.assertIs(step.fit(self.dataset), step)

        actual = step.model.get_params()
        for name, expected in parameters.items():
            with self.subTest(parameter=name):
                self.assertEqual(actual[name], expected)

        predictions = np.asarray(step.predict(self.dataset.X))
        self.assertEqual(predictions.shape, (len(self.dataset.X),))
        self.assertTrue(np.isfinite(predictions).all())
        return step.model

    def test_random_survival_forest_uses_configured_parameters(self):
        """Seven requested trees must produce seven fitted trees with bounded depth."""
        model = self._fit_configured(ActRandomSurvivalForest, {
            "n_estimators": 7,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_depth": 2,
        })
        self.assertEqual(len(model.estimators_), 7)
        for tree in model.estimators_:
            self.assertLessEqual(tree.tree_.max_depth, 2)

    def test_extra_survival_trees_uses_configured_parameters(self):
        """Randomness, feature selection and tree limits all reach ExtraSurvivalTrees."""
        model = self._fit_configured(ActExtraSurvivalTrees, {
            "n_estimators": 7,
            "max_depth": 2,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
            "max_features": 1,
            "random_state": 41,
        })
        self.assertEqual(len(model.estimators_), 7)
        for tree in model.estimators_:
            self.assertLessEqual(tree.tree_.max_depth, 2)

    def test_cox_uses_configured_parameters(self):
        """Regularization, ties and optimization settings reach the fitted Cox model."""
        self._fit_configured(ActCox, {
            "alpha": 2.5,
            "ties": "efron",
            "n_iter": 37,
            "tol": 1e-6,
        })

    def test_survival_tree_uses_configured_parameters(self):
        """The standalone tree honors every exposed parameter, including low-memory mode."""
        model = self._fit_configured(ActSurvivalTree, {
            "splitter": "random",
            "max_depth": 2,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "min_weight_fraction_leaf": 0.05,
            "max_features": 1,
            "random_state": 41,
            "max_leaf_nodes": 4,
            "low_memory": True,
        })
        self.assertLessEqual(model.tree_.max_depth, 2)
        self.assertLessEqual(model.tree_.n_leaves, 4)

    def _check_gradient_boosting(self, step_class):
        model = self._fit_configured(step_class, {
            "n_estimators": 7,
            "learning_rate": 0.3,
            "max_depth": 2,
            "min_samples_split": 8,
            "min_samples_leaf": 3,
        })
        self.assertEqual(len(model.estimators_), 7)
        for tree in model.estimators_.ravel():
            self.assertLessEqual(tree.tree_.max_depth, 2)

    def test_gradient_boosting_uses_configured_parameters(self):
        """The regular gradient-boosting step honors its configured stages and trees."""
        self._check_gradient_boosting(ActGradientBoostingSurvivalAnalysis)

    def test_legacy_survival_xgboost_import_uses_configured_parameters(self):
        """The legacy import retains configurable parameters on the canonical model."""
        self._check_gradient_boosting(LegacyGradientBoostingSurvivalAnalysis)

    def test_survival_xgboost_uses_configured_parameters(self):
        """Every exposed Cox boosting option reaches the real XGBoost estimator."""
        model = self._fit_configured(ActSurvivalXGBoost, {
            "n_estimators": 7, "max_depth": 2, "learning_rate": 0.3,
            "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 2.5,
            "reg_alpha": 0.2, "reg_lambda": 3.0, "gamma": 0.1, "random_state": 41,
        })
        self.assertEqual(model.get_booster().num_boosted_rounds(), 7)
        actual = model.get_params()
        self.assertEqual(actual["objective"], "survival:cox")
        self.assertEqual(actual["tree_method"], "hist")
        self.assertEqual(actual["n_jobs"], 1)

    def test_componentwise_boosting_uses_configured_parameters(self):
        """Componentwise boosting fits the requested number of stages and learning rate."""
        model = self._fit_configured(ActComponentwiseGradientBoostingSurvivalAnalysis, {
            "n_estimators": 7,
            "learning_rate": 0.3,
        })
        self.assertEqual(len(model.estimators_), 7)

    def test_declared_defaults_match_fitted_models(self):
        """IAML defaults, including forest leaf sizes and Cox alpha, are actually applied."""
        for step_class in SURVIVAL_STEPS:
            with self.subTest(step=step_class.__module__):
                step = step_class()
                step.fit(self.dataset)
                actual = step.model.get_params()
                for name, specification in step.configuration.items():
                    with self.subTest(parameter=name):
                        expected = specification["default"]
                        if name == "low_memory" and expected == "auto":
                            expected = False  # An unspecified output contract preserves curves.
                        self.assertEqual(actual[name], expected)

    def test_invalid_number_of_estimators_is_rejected(self):
        """Invalid values reach backend validation instead of silently using 100 stages."""
        for step_class in ENSEMBLE_STEPS:
            with self.subTest(step=step_class.__module__):
                step = step_class()
                step.configure({"n_estimators": 0})
                with self.assertRaisesRegex(ValueError, "n_estimators"):
                    step.fit(self.dataset)

    def test_categorical_search_choices_are_accepted_by_estimators(self):
        """Every advertised categorical choice must be usable in a real fit."""
        for step_class, names in (
            (ActExtraSurvivalTrees, ("max_features",)),
            (ActSurvivalTree, ("splitter", "max_features", "low_memory")),
        ):
            configuration = step_class().configuration
            for name in names:
                for value in configuration[name]["categorical"]:
                    with self.subTest(step=step_class.__name__, parameter=name, value=value):
                        parameters = {name: value}
                        if "n_estimators" in configuration:
                            parameters["n_estimators"] = 7
                        self._fit_configured(step_class, parameters)

    def test_componentwise_boosting_rejects_tree_parameters(self):
        """A model with linear base learners must not advertise ineffective tree options."""
        for name in ("max_depth", "min_samples_split", "min_samples_leaf"):
            with self.subTest(parameter=name):
                step = ActComponentwiseGradientBoostingSurvivalAnalysis()
                with self.assertRaisesRegex(AttributeError, name):
                    step.configure({name: 2})


if __name__ == "__main__":
    unittest.main()
