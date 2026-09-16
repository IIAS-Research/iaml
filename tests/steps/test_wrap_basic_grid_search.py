"""Grid search must produce independent candidates through the real step API."""
from copy import deepcopy
from itertools import product

import numpy as np
from sklearn.preprocessing import normalize

from .step_test_case import StepTestCase

from iaml.actionables.normalize.act_normalizer import ActNormalizer
from iaml.actionables.normalize.act_standard_scaler import ActStandardScaler
from iaml.actionables.predictors.classifier.act_decision_tree_classifier import (
    ActDecisionTreeClassifier,
)
from iaml.actionables.predictors.classifier.act_randomforest import ActRandomForest
from iaml.candidate import Candidate
from iaml.wrapper.wrap_basic_gridsearch import WrapBasicGridSearch
from tests.helpers.datasets import make_classification_data


class TestWrapBasicGridSearch(StepTestCase):
    def make_candidate(self):
        return Candidate(self.make_dataset(*make_classification_data(n_samples=20)))

    def test_bounded_numeric_and_categorical_grid_contains_every_combination(self):
        step = ActDecisionTreeClassifier()
        step.configure({"max_depth": 10, "min_samples_leaf": 2, "random_state": 123})
        step.configuration["max_depth"]["range"] = [8, 12]
        for name, item in step.configuration.items():
            if name not in {"max_depth", "min_samples_leaf", "criterion", "random_state"}:
                item["no_gridsearch"] = True
        original_configuration = deepcopy(step.configuration)
        candidate = self.make_candidate()

        results = WrapBasicGridSearch(step).run(candidate)

        expected = set(product(range(8, 13), [1, 2, 3], ["gini", "entropy", "log_loss"]))
        predictors = [result.pipeline.predictor[1] for result in results]
        self.assertEqual(len(results), len(expected))
        self.assertEqual({
            (model.get_config("max_depth"), model.get_config("min_samples_leaf"),
             model.get_config("criterion")) for model in predictors
        }, expected)
        self.assertTrue(all(model.get_config("random_state") == 123 for model in predictors))
        self.assertTrue(all(model.get_config("max_features") == 1.0 for model in predictors))
        self.assertEqual(len({id(model) for model in predictors}), len(results))
        self.assertEqual(step.configuration, original_configuration)
        self.assertFalse(candidate.pipeline.have_model)
        predictors[0].fit(candidate.dataset)
        self.assertEqual(predictors[0].model.max_depth, predictors[0].get_config("max_depth"))

    def test_boolean_values_are_both_explored(self):
        step = ActRandomForest()
        step.configure("random_state", 123)
        for name, item in step.configuration.items():
            if name not in {"bootstrap", "random_state"}:
                item["no_gridsearch"] = True

        results = WrapBasicGridSearch(step).run(self.make_candidate())

        self.assertEqual(len(results), 2)
        predictors = [result.pipeline.predictor[1] for result in results]
        self.assertEqual({model.get_config("bootstrap") for model in predictors}, {True, False})
        self.assertTrue(all(type(model.get_config("bootstrap")) is bool for model in predictors))
        self.assertTrue(all(model.get_config("random_state") == 123 for model in predictors))
        self.assertFalse(step.get_config("bootstrap"))

    def test_each_transform_starts_from_the_original_candidate(self):
        step = ActNormalizer()
        candidate = self.make_candidate()
        original = candidate.dataset.X.copy()

        results = WrapBasicGridSearch(step).run(candidate)

        self.assertEqual(len(results), 4)
        configurations = set()
        for result in results:
            self.assertEqual(len(result.pipeline.transformers), 1)
            transform = result.pipeline.transformers[0][1]
            configurations.add((transform.get_config("norm"), transform.get_config("copy")))
            np.testing.assert_allclose(
                result.dataset.X, normalize(original, norm=transform.get_config("norm")),
            )
        self.assertEqual(configurations, set(product(["l1", "l2"], [True, False])))
        self.assertFrameEqual(candidate.dataset.X, original)
        self.assertEqual(candidate.pipeline.transformers, [])
        self.assertEqual(step.resume_configuration(), {"norm": "l2", "copy": True})

    def test_step_without_parameters_runs_once(self):
        candidate = self.make_candidate()
        original = candidate.dataset.X.copy()

        results = WrapBasicGridSearch(ActStandardScaler()).run(candidate)

        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0].pipeline.transformers), 1)
        expected = (original - original.mean()) / original.std(ddof=0)
        self.assertFrameEqual(results[0].dataset.X, expected)
        self.assertFrameEqual(candidate.dataset.X, original)
        self.assertEqual(candidate.pipeline.transformers, [])
