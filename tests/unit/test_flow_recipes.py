"""Behavioral contracts for declarative recipe editing and reconstruction."""
import pickle
import unittest
from types import SimpleNamespace
from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval

from iaml import Step
from iaml.flow import (Const, Float, Int, PipelineSpec, choice, metrics, optional,
                       normalizers, statistics, use)
from iaml.steps import (DecisionTreeClassifier, RandomForestClassifier, StandardScaler,
                        RobustScaler, MinMaxScaler, SMOTE, SMOTETomek, SMOTEENN,
                        SimpleImputer, Normalizer, UnitNormScaler)
from iaml.metrics import RecallMetric
from iaml.statistics import TopKValueCountsStatistic


class FlowRecipeTests(unittest.TestCase):
    def forest(self, alias, low, high):
        return use(RandomForestClassifier, n_estimators=Int(low, high, initial=low)).named(alias)

    def test_composition_and_constructor_copy_fragments(self):
        forest = self.forest("forest", 100, 500)
        models = choice(forest)
        pipeline = use(SimpleImputer) >> models
        forest.configure(n_estimators=150)
        self.assertEqual(pipeline["forest"].parameters["n_estimators"].value, 100)
        pipeline["forest"].configure(n_estimators=200)
        self.assertEqual(models["forest"].parameters["n_estimators"].value, 100)

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

    def test_selection_is_atomic_and_keeps_variant_domains(self):
        models = choice(self.forest("small", 10, 100), self.forest("big", 100, 500))
        selection = models.find_all(RandomForestClassifier)
        with self.assertRaises(ValueError):
            selection.configure(n_estimators=200)
        self.assertEqual(models["small"].parameters["n_estimators"].value, 10)
        selection.configure(n_estimators=100)
        self.assertEqual(models["small"].parameters["n_estimators"].domain, (10, 100))
        models.add(self.forest("new", 1, 500))
        selection.configure(n_estimators=100)
        self.assertEqual(models["new"].parameters["n_estimators"].value, 1)
        models.remove("small")
        with self.assertRaises(ValueError):
            selection.configure(n_estimators=100)

    def test_aliases_and_replace_are_atomic_and_detach_old_reference(self):
        pipeline = use(SimpleImputer).named("cleaning") >> choice(StandardScaler).named("normalize")
        old = pipeline.normalize
        replacement = old.replace(choice(RobustScaler))
        self.assertIs(pipeline.normalize, replacement)
        self.assertIsNone(old._parent)
        old.add(StandardScaler)
        self.assertEqual(len(list(pipeline.normalize)), 1)
        with self.assertRaises(ValueError):
            replacement.replace(choice(StandardScaler).named("different"))
        with self.assertRaises(ValueError):
            pipeline.add(use(StandardScaler).named("cleaning"))
        with self.assertRaises(TypeError):
            pipeline[StandardScaler]

    def test_root_replace_and_insertion_into_default_main(self):
        owner = SimpleNamespace(pipeline=PipelineSpec.default())
        owner.pipeline.attach(owner, "pipeline")
        owner.pipeline.add(use(StandardScaler).named("extra"), before="cleaning")
        sequence = list(owner.pipeline.main)
        self.assertLess([n.alias for n in sequence].index("extra"), [n.alias for n in sequence].index("cleaning"))
        owner.pipeline.remove("imbalance")
        with self.assertRaises(KeyError):
            owner.pipeline["imbalance"]
        old = owner.pipeline
        new = old.replace(use(RandomForestClassifier))
        self.assertIs(owner.pipeline, new)
        self.assertIsNone(old._owner)

    def test_start_and_strict_subtraction(self):
        models = choice(self.forest("one", 1, 100), self.forest("two", 1, 500)).start("one")
        with self.assertRaises(ValueError):
            models.remove("one")
        models.start().remove("one")
        with self.assertRaises(ValueError):
            models.remove(StandardScaler)
        family = normalizers().remove(Normalizer)
        with self.assertRaises(ValueError):
            family.remove(Normalizer)
        family.add(use(Normalizer).named("manual"))
        self.assertEqual(len(family.find_all(Normalizer)), 1)
        self.assertIn(Normalizer, family.excluded)

    def test_unit_norm_scaler_preserves_family_scope_and_reconstruction(self):
        family = normalizers()
        self.assertEqual(len(family.find_all(UnitNormScaler)), 1)
        family.remove(UnitNormScaler)
        self.assertEqual(len(family.find_all(Normalizer)), 0)
        with self.assertRaises(ValueError):
            family.remove(Normalizer)

        family.add(use(UnitNormScaler, norm=Const("l1")).named("unit_norm"))
        self.assertIs(family["unit_norm"].component, Normalizer)
        self.assertEqual(family["unit_norm"].instantiate().get_config("norm"), "l1")
        self.assertIn(Normalizer, family.excluded)

        namespace = {}
        code = family.to_code()
        self.assertIn("import ActUnitNormScaler as", code)
        component_name = family["unit_norm"].describe()["tree"]["component"]
        self.assertTrue(component_name.endswith(".ActUnitNormScaler"))
        exec(code, namespace)
        restored = namespace["pipeline"]
        self.assertEqual(family.describe().to_dict(), restored.describe().to_dict())

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

    def test_renaming_a_starting_alternative_preserves_its_reference(self):
        models = choice(self.forest("one", 1, 100), self.forest("two", 1, 500)).start("one")
        models["one"].named("renamed")
        self.assertEqual(models.initial_aliases, ("renamed",))
        with self.assertRaises(ValueError):
            models["renamed"].named("two")
        self.assertEqual(models.initial_aliases, ("renamed",))

    def test_analytic_configuration_is_fixed_and_kind_checked(self):
        analyses = metrics(use(RecallMetric, pos_label=0).named("negative"))
        self.assertEqual(analyses.kind, "metrics")
        self.assertEqual(analyses["negative"].instantiate().pos_label, 0)
        with self.assertRaises(TypeError):
            analyses["negative"].configure(pos_label=Int(0, 1, initial=0))
        with self.assertRaises(TypeError):
            analyses.add(TopKValueCountsStatistic)
        description = statistics(use(TopKValueCountsStatistic, k=5))
        self.assertEqual(description.resolved()[0].instantiate().k, 5)

    def test_training_and_analysis_recipes_cannot_be_mixed(self):
        with self.assertRaises(TypeError):
            PipelineSpec([use(RecallMetric)])
        with self.assertRaises(TypeError):
            use(SimpleImputer) >> metrics(RecallMetric)
        with self.assertRaises(TypeError):
            metrics(RandomForestClassifier)
        with self.assertRaises(TypeError):
            optional(use(RecallMetric))

    def test_to_code_preserves_domains_aliases_and_partial_start(self):
        models = choice(self.forest("forest", 100, 500), use(RandomForestClassifier).named("other"))
        models["forest"].configure(n_estimators=Const(700))
        models.start("forest")
        pipeline = use(SimpleImputer) >> models
        namespace = {}
        exec(pipeline.to_code(), namespace)
        restored = namespace["pipeline"]
        self.assertEqual(pipeline.describe().to_dict(), restored.describe().to_dict())

    def test_default_reconstruction_keeps_subtractions_and_variant_edits(self):
        pipeline = PipelineSpec.default()
        pipeline.normalize.remove(Normalizer)
        next(iter(pipeline.normalize.find_all(StandardScaler))).named("standard")
        pipeline.normalize.start("standard")
        pipeline.predictor.add(self.forest("small", 10, 100))
        pipeline["small"].configure(n_estimators=Const(70))
        pipeline.remove("imbalance")
        namespace = {}
        exec(pipeline.to_code(), namespace)
        self.assertEqual(pipeline.describe().to_dict(), namespace["pipeline"].describe().to_dict())

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

    def test_selection_detects_detached_group_and_root(self):
        pipeline = use(SimpleImputer) >> choice(RandomForestClassifier).named("models")
        selection = pipeline.models.find_all(RandomForestClassifier)
        pipeline.models.replace(choice(RandomForestClassifier))
        with self.assertRaises(ValueError):
            selection.configure(n_estimators=200)
        owner = SimpleNamespace(pipeline=use(RandomForestClassifier))
        owner.pipeline.attach(owner, "pipeline")
        selection = owner.pipeline.find_all(RandomForestClassifier)
        owner.pipeline.replace(use(RandomForestClassifier))
        with self.assertRaises(ValueError):
            selection.configure(n_estimators=200)

    def test_reconstruction_preserves_edited_optional_and_valid_inactive_start(self):
        # A search interval can be wider than intrinsic estimator constraints;
        # its actual start must be valid and remain valid during reconstruction.
        forest = use(RandomForestClassifier, n_estimators=Int(0, 5, initial=2))
        forest.configure(n_estimators=Const(700))
        pipeline = optional(use(StandardScaler)).add(use(RobustScaler)) >> forest
        namespace = {}
        exec(pipeline.to_code(), namespace)
        self.assertEqual(pipeline.describe().to_dict(), namespace["pipeline"].describe().to_dict())

    def test_analytic_component_replacement_enforces_container_contract_atomically(self):
        for analyses, replacement in (
            (metrics(use(RecallMetric).named("item")), metrics(RecallMetric)),
            (statistics(use(TopKValueCountsStatistic).named("item")),
             statistics(TopKValueCountsStatistic)),
        ):
            with self.subTest(kind=analyses.kind):
                old = analyses["item"]
                before = analyses.describe().to_dict()
                with self.assertRaises(TypeError):
                    old.replace(replacement)
                self.assertIs(analyses["item"], old)
                self.assertIs(old._parent, analyses)
                self.assertEqual(analyses.describe().to_dict(), before)
                new = old.replace(use(old.component))
                self.assertIs(analyses["item"], new)
                self.assertIsNone(old._parent)

    def test_constants_follow_backend_union_types_and_reject_invalid_values(self):
        for key, value in (("max_depth", None), ("min_samples_leaf", 0.1),
                           ("min_samples_split", 0.5)):
            with self.subTest(parameter=key):
                recipe = use(DecisionTreeClassifier, **{key: Const(value)})
                self.assertEqual(recipe.instantiate().get_config(key), value)
                self.assertTrue(recipe.parameters[key].fixed)
                recipe.configure(**{key: Const(2)})
                recipe.configure(**{key: Const(value)})
                before = recipe.describe().to_dict()
                with self.assertRaises(ValueError):
                    recipe.configure(**{key: Const(-1)})
                self.assertEqual(recipe.describe().to_dict(), before)
        with self.assertRaises(ValueError):
            use(DecisionTreeClassifier, min_samples_leaf=0.1)
        with self.assertRaises(TypeError):
            use(DecisionTreeClassifier, max_depth=Const(True))
        self.assertEqual(use(RandomForestClassifier, n_estimators=Const(np.int64(10)))
                         .instantiate().get_config("n_estimators"), 10)

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

    def test_composite_resampler_constants_are_validated_before_instantiation(self):
        for component in (SMOTE, SMOTETomek, SMOTEENN):
            with self.subTest(component=component.__name__):
                recipe = use(component)
                before = recipe.describe().to_dict()
                with self.assertRaises(ValueError):
                    recipe.configure(sampling_strategy=Const("nonsense"))
                self.assertEqual(recipe.describe().to_dict(), before)
                with self.assertRaises(ValueError):
                    use(component, k_neighbors=Const(0))
                self.assertEqual(use(component, sampling_strategy=Const(0.75))
                                 .instantiate().get_config("sampling_strategy"), 0.75)
        with self.assertRaises(ValueError):
            use(SMOTEENN, kind_sel=Const("nonsense"))

    def test_reconstruction_keeps_single_child_and_nested_sequence_containers(self):
        from iaml.flow.model import SequenceSpec
        recipe = (use(StandardScaler).named("scale")
                  >> use(DecisionTreeClassifier).named("tree")).named("sequence")
        recipe.remove("scale")
        namespace = {}
        exec(recipe.to_code(), namespace)
        restored = namespace["pipeline"]
        self.assertEqual(recipe.describe().to_dict(), restored.describe().to_dict())
        restored.add(use(StandardScaler).named("new"), before="tree")
        self.assertEqual([node.alias for node in restored], ["new", "tree"])

        nested = SequenceSpec([use(StandardScaler) >> use(RobustScaler),
                               use(DecisionTreeClassifier)])
        namespace = {}
        exec(nested.to_code(), namespace)
        self.assertEqual(nested.describe().to_dict(), namespace["pipeline"].describe().to_dict())

    def test_family_replacement_retains_slot_and_reconstructs_named_or_unnamed_variants(self):
        for alias in (None, "scale"):
            with self.subTest(alias=alias):
                family = normalizers()
                old = next(iter(family.find_all(MinMaxScaler)))
                if alias:
                    old.named(alias)
                index = list(family).index(old)
                selection = family.find_all(MinMaxScaler)
                replacement = old.replace(use(RobustScaler))
                self.assertIs(list(family)[index], replacement)
                self.assertIsNone(old._parent)
                with self.assertRaises(ValueError):
                    selection.configure()
                namespace = {}
                exec(family.to_code(), namespace)
                self.assertEqual(family.describe().to_dict(), namespace["pipeline"].describe().to_dict())
                self.assertEqual(family.describe().to_dict(), family.clone().describe().to_dict())

    def test_family_replacements_remain_explicit_across_exclusions_and_removal(self):
        family = normalizers().remove(RobustScaler)
        old = next(iter(family.find_all(StandardScaler))).named("custom")
        old.replace(use(RobustScaler))
        # Removing another explicit variant of the original class must not
        # conceal the replacement sitting in that class's registry slot.
        family.add(use(StandardScaler).named("extra")).remove(StandardScaler)
        self.assertEqual(len(family.find_all(RobustScaler)), 1)
        namespace = {}
        exec(family.to_code(), namespace)
        self.assertEqual(family.describe().to_dict(), namespace["pipeline"].describe().to_dict())
        family.remove("custom")
        self.assertEqual(len(family.find_all(RobustScaler)), 0)
        namespace = {}
        exec(family.to_code(), namespace)
        self.assertEqual(family.describe().to_dict(), namespace["pipeline"].describe().to_dict())

    def test_family_removal_reconstruction_allows_reusing_the_removed_alias(self):
        family = normalizers()
        next(iter(family.find_all(StandardScaler))).named("reused")
        family.remove("reused")
        next(iter(family.find_all(RobustScaler))).named("reused")
        namespace = {}
        exec(family.to_code(), namespace)
        self.assertEqual(family.describe().to_dict(), namespace["pipeline"].describe().to_dict())

    def test_chained_family_replacements_keep_group_modes_and_detached_references(self):
        family = normalizers()
        original = next(iter(family.find_all(MinMaxScaler))).named("slot")
        index = list(family).index(original)
        group = original.replace(choice(StandardScaler, RobustScaler))
        namespace = {}
        exec(family.to_code(), namespace)
        self.assertEqual(family.describe().to_dict(), namespace["pipeline"].describe().to_dict())
        replacement = group.replace(use(UnitNormScaler, norm=Const("l1")))
        self.assertIs(list(family)[index], replacement)
        self.assertIsNone(group._parent)
        group.add(MinMaxScaler)
        self.assertIs(family["slot"], replacement)
        family.remove(UnitNormScaler)
        self.assertEqual(len(family.find_all(UnitNormScaler)), 0)
        namespace = {}
        exec(family.to_code(), namespace)
        self.assertEqual(family.describe().to_dict(), namespace["pipeline"].describe().to_dict())


if __name__ == "__main__":
    unittest.main()
