"""Compilation exercises the real branch generation and local search metadata."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from iaml import Candidate, Dataset, Step
from iaml.flow import (Const, Int, PipelineSpec, choice, compile_pipeline,
                       normalizers, optional, use)
from iaml.flow.compiler import RecipeValidationError
from iaml.steps import (ADASYN, DecisionTreeClassifier, DecisionTreeRegressor, LogisticRegression,
                        RandomForestClassifier, RobustScaler, SimpleImputer,
                        SMOTE, StandardScaler)
from iaml.optimizers import GeneticOptimizer, RandomOptimizer
from iaml.void_step import VoidStep


class FlowCompilerTests(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(pd.DataFrame({"a": np.arange(16, dtype=float),
                                             "b": np.arange(16, dtype=float) ** 2}),
                               np.array([0, 1] * 8))

    def test_full_pipeline_generates_four_real_branches(self):
        spec = (choice(StandardScaler, RobustScaler)
                >> choice(DecisionTreeClassifier, RandomForestClassifier))
        root = compile_pipeline(spec)
        outputs = root.run(Candidate(self.dataset))
        self.assertEqual(len(outputs), 4)
        for candidate in outputs:
            self.assertEqual(len(candidate.pipeline.training_steps), 2)

    def test_partial_start_keeps_all_local_variants(self):
        models = choice(use(DecisionTreeClassifier, max_depth=Const(2)).named("tiny"),
                        use(DecisionTreeClassifier, max_depth=Int(3, 10, initial=5)).named("large"))
        models.start("tiny")
        root = compile_pipeline(models)
        self.assertEqual(len(root.steps), 1)
        step = root.steps[0]
        self.assertEqual([s._flow_alias for s in step._flow_alternatives], ["tiny", "large"])
        self.assertEqual(step._flow_alternatives[1].configuration["max_depth"]["range"], [3, 10])
        self.assertTrue(all(not hasattr(s, "_flow_alternatives") for s in step._flow_alternatives))
        with self.assertRaises(ValueError):
            compile_pipeline(models, optimizer=RandomOptimizer)

    def test_optional_has_explicit_absence_and_non_optional_does_not(self):
        root = compile_pipeline(optional(use(StandardScaler)) >> use(DecisionTreeClassifier))
        self.assertEqual(len(root.steps[0].steps), 2)
        self.assertIsInstance(root.steps[0].steps[-1], VoidStep)
        outputs = root.run(Candidate(self.dataset))
        self.assertEqual(len(outputs), 2)
        model = compile_pipeline(use(DecisionTreeClassifier))
        self.assertFalse(model.is_interchangeable)

    def test_inapplicable_required_choice_rejects_its_paths(self):
        regression = Dataset(self.dataset.X, np.arange(16) * 0.17)
        root = compile_pipeline(choice(SMOTE, ADASYN) >> use(DecisionTreeRegressor))
        self.assertEqual(root.run(Candidate(regression)), [])

    def test_inapplicable_alternative_keeps_only_the_declared_compatible_path(self):
        regression = Dataset(self.dataset.X, np.arange(16) * 0.17)
        root = compile_pipeline(choice(SMOTE, StandardScaler) >> use(DecisionTreeRegressor))
        outputs = root.run(Candidate(regression))
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0].pipeline.transformers[0][1], StandardScaler)
        self.assertEqual(len(outputs[0].pipeline.training_steps), 2)

    def test_optional_inapplicable_component_keeps_only_the_explicit_absence(self):
        regression = Dataset(self.dataset.X, np.arange(16) * 0.17)
        root = compile_pipeline(optional(use(SMOTE)) >> use(DecisionTreeRegressor))
        outputs = root.run(Candidate(regression))
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0].pipeline.resamplers[0][1], VoidStep)

    def test_complete_strategy_cannot_skip_its_declared_preparation(self):
        regression = Dataset(self.dataset.X, np.arange(16) * 0.17)
        root = compile_pipeline(choice(
            use(SMOTE) >> use(DecisionTreeRegressor),
            use(StandardScaler) >> use(DecisionTreeRegressor),
        ))
        outputs = root.run(Candidate(regression))
        self.assertEqual(len(outputs), 1)
        self.assertIsInstance(outputs[0].pipeline.transformers[0][1], StandardScaler)

    def test_single_required_alternative_is_still_required_when_refitted(self):
        root = compile_pipeline(choice(StandardScaler) >> use(DecisionTreeClassifier))
        candidate = root.run(Candidate(self.dataset))[0]
        scaler = candidate.pipeline.transformers[0][1]
        self.assertFalse(scaler.is_interchangeable)
        with patch.object(scaler, 'suitable', return_value=False):
            with self.assertRaisesRegex(ValueError, 'Required pipeline step.*inapplicable'):
                candidate.pipeline.fit_transform(self.dataset.X, self.dataset.y)

    def test_complete_strategies_cannot_swap_their_models(self):
        linear = use(StandardScaler) >> use(LogisticRegression)
        root = compile_pipeline(choice(linear, use(RandomForestClassifier)))
        self.assertFalse(root.steps[1].is_interchangeable)
        self.assertFalse(root.steps[0].steps[1].is_interchangeable)
        with self.assertRaises(ValueError):
            compile_pipeline(choice(linear.named("linear"), use(RandomForestClassifier).named("forest")).start("linear"))

    def test_structure_is_validated_before_runtime_generation(self):
        for spec in (use(StandardScaler), choice(use(DecisionTreeClassifier), use(StandardScaler)),
                     use(DecisionTreeClassifier) >> use(StandardScaler),
                     use(DecisionTreeClassifier) >> use(DecisionTreeClassifier),
                     optional(use(DecisionTreeClassifier))):
            with self.subTest(recipe=spec):
                with self.assertRaises(RecipeValidationError):
                    compile_pipeline(spec)

    def test_structural_error_is_distinct_from_optimizer_capability_error(self):
        with self.assertRaises(RecipeValidationError):
            compile_pipeline(choice())
        spec = choice(use(DecisionTreeClassifier).named("tree"),
                      use(RandomForestClassifier).named("forest")).start("tree")
        try:
            compile_pipeline(spec, optimizer=RandomOptimizer)
        except RecipeValidationError:
            self.fail("Optimizer capability errors must not be deferred as incomplete recipes")
        except ValueError:
            pass
        else:
            self.fail("RandomOptimizer must reject partial component starts")

    def test_compile_freezes_family_and_never_mutates_recipe(self):
        recipe = normalizers() >> use(DecisionTreeClassifier)
        first = compile_pipeline(recipe)
        count = len(first.steps[0].steps)
        from iaml.decorators.is_step import is_step
        @is_step("normalize")
        class NewNormalizer(StandardScaler):
            pass
        self.addCleanup(Step.available_steps.pop, NewNormalizer, None)
        self.assertEqual(len(first.steps[0].steps), count)
        second = compile_pipeline(recipe)
        self.assertEqual(len(second.steps[0].steps), count + 1)
        self.assertFalse(recipe._frozen)






if __name__ == "__main__":
    unittest.main()
