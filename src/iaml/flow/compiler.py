"""Compile recipe snapshots to IAML's existing execution tree."""
from __future__ import annotations

from copy import deepcopy

from ..step import Step
from ..metastep import MetaStep
from ..meta_ordered_step import MetaOrderedStep
from ..meta_explorer_step import MetaExplorerStep
from ..meta_partial_explorer_step import MetaPartialExplorerStep
from ..void_step import VoidStep
from ..decorators.is_step import is_step
from ..decorators.runner import runner
from ..optimizers.genetic_optimizer import GeneticOptimizer
from .model import (AdaptiveSpec, ChoiceSpec, ComponentSpec, GroupSpec,
                    PipelineSpec, SequenceSpec, choice, use)


class RecipeValidationError(ValueError):
    """A recipe's current structure is incomplete or invalid for training.

    Study construction may defer this error while the recipe is being edited;
    parameter errors and optimizer capability errors are separate failures.
    """


@is_step("meta")
class _PresetExplorerStep(MetaExplorerStep):
    """Generate declared strategies in order, preserving baseline warmup."""
    @runner
    def run(self, candidate):
        output = []
        for step in self.steps:
            output.extend(step.run(candidate.to_input()))
        return output


@is_step("meta")
class _ConditionalImputerStep(Step):
    """Preserve the preset's generation-time missingness condition."""
    def __init__(self, component):
        self.component = component

    @runner
    def run(self, candidate):
        if candidate.dataset.X.isna().values.any():
            return self.component.run(candidate)
        return candidate

    def all_steps(self):
        return [self, self.component]

    def configure_parents(self, *parents):
        self.component.configure_parents(*parents)
        super().configure_parents(*parents)


def supports_component_swaps(optimizer):
    return (isinstance(optimizer, type) and issubclass(optimizer, GeneticOptimizer)
            or bool(getattr(optimizer, "supports_component_swap", False)))


def default_spec():
    """Build one explicit preset shared by IAML and PipelineSpec.default."""
    from ..actionables.cleaning.act_simple_imputer import ActSimpleImputer
    main = SequenceSpec([
        AdaptiveSpec(tag="features_precleaning").named("features_precleaning"),
        AdaptiveSpec(tag="cleaning").named("cleaning"),
        AdaptiveSpec(tag="features_selection").named("features_selection"),
        choice(tag="normalize").named("normalize"),
        choice(tag="imbalance").named("imbalance"),
        choice(tag="features_preprocessing").named("features_preprocessing"),
        choice(tag="predictor").named("predictor"),
    ]).named("main")
    main.normalize._preset_start = "standard"
    main.normalize._allow_absence = True
    main.imbalance._preset_start = "absence"
    main.imbalance._allow_absence = True
    main.features_preprocessing._preset_start = "preprocessing"
    main.features_preprocessing._allow_absence = True
    imputer = use(ActSimpleImputer).named("minimal_imputer")
    imputer._conditional_missing = True
    minimal = SequenceSpec([
        imputer, choice(tag="minimal_predictor").named("minimal_predictor"),
    ]).named("minimal")
    # The old implementation generated this branch first and warms up the first
    # resulting candidate after both pools have been generated.
    preset = PipelineSpec([minimal, main])
    preset._is_default_preset = True
    return preset


def _path_shapes(node):
    """Structural predictor checks without eagerly enumerating the product."""
    if isinstance(node, ComponentSpec):
        is_model = issubclass(node.component, __import__("iaml.predictor", fromlist=["Predictor"]).Predictor)
        return {(1, True) if is_model else (0, False)}
    children = node._children()
    if not children:
        if isinstance(node, ChoiceSpec):
            raise RecipeValidationError(f"Choice {node.alias or '<unnamed>'} has no alternatives")
        return {(0, False)}
    if isinstance(node, ChoiceSpec):
        shapes = set().union(*(_path_shapes(child) for child in children))
        if node._allow_absence:
            if any(count for count, _ in shapes):
                raise RecipeValidationError("optional cannot remove a predictor")
            shapes.add((0, False))
        return shapes
    shapes = {(0, False)}
    for child in children:
        next_shapes = set()
        for count, terminal in shapes:
            for child_count, child_terminal in _path_shapes(child):
                if terminal:
                    raise RecipeValidationError("A predictor must be terminal on every candidate path")
                next_shapes.add((count + child_count, child_terminal))
        shapes = next_shapes
    return shapes


def _void_template(representative, choice_id):
    void = VoidStep(step_to_mimic=deepcopy(representative))
    void._flow_explicit = True
    void._flow_parameters = {}
    void._flow_choice_id = choice_id
    void._flow_node_id = f"{choice_id}:absence"
    void._flow_variant_id = f"{choice_id}:absence"
    void._flow_alias = None
    return void


def _compile(node, optimizer, preprocessor, fast, *, required=False):
    if isinstance(node, ComponentSpec):
        component = node.instantiate()
        component._flow_required = required
        if getattr(node, "_conditional_missing", False):
            return _ConditionalImputerStep(component)
        return component
    if isinstance(node, ChoiceSpec):
        alternatives = node._children()
        if fast and node.alias == "predictor" and getattr(node, "tag", None) == "predictor":
            alternatives = [item for item in alternatives if "fast_predictor" in Step.available_steps.get(item.component, ())]
        if not alternatives:
            raise ValueError(f"Choice {node.alias or '<unnamed>'} has no alternatives")
        simple = all(isinstance(child, ComponentSpec) and not getattr(child, "_conditional_missing", False)
                     for child in alternatives)
        partial = node.initial_aliases is not None
        if partial and (not simple or not supports_component_swaps(optimizer)):
            raise ValueError("Partial start requires simple components and an optimizer supporting component swaps")
        # Inapplicable alternatives must be rejected as paths, rather than
        # passing through an input without the declared operation. Optional
        # choices generate their absence as a separate, explicit alternative.
        templates = [_compile(child, optimizer, preprocessor, fast, required=True)
                     for child in alternatives]
        for index, template in enumerate(templates):
            template._flow_choice_id = node.node_id
            template._flow_variant_key = index
            template.is_interchangeable = simple
        if node._allow_absence and simple:
            void = _void_template(templates[0], node.node_id)
            void._flow_variant_key = len(templates)
            templates.append(void)
        initial = templates
        if partial:
            initial = [template for template in templates if template._flow_alias in node.initial_aliases]
        elif node._preset_start and simple:
            if node._preset_start == "standard" and supports_component_swaps(optimizer):
                from ..actionables.normalize.act_standard_scaler import ActStandardScaler
                initial = [template for template in templates if isinstance(template, ActStandardScaler)]
                if not initial:
                    initial = templates[:1]
            elif node._preset_start == "standard":
                # The historical non-mutating optimizers explore every scaler
                # initially; absence is part of the preset's mutation policy.
                initial = templates[:-1] if node._allow_absence else templates
            elif node._preset_start == "absence" and supports_component_swaps(optimizer):
                initial = templates[-1:]
            elif node._preset_start == "preprocessing" and not preprocessor:
                initial = templates[-1:]
        if not initial:
            raise ValueError(f"No initial alternative for choice {node.alias or '<unnamed>'}")
        group_type = _PresetExplorerStep if getattr(node, "_is_default_preset", False) else MetaExplorerStep
        if simple and len(initial) == 1 and len(templates) > 1:
            group_type = MetaPartialExplorerStep
        group = group_type(name=node.alias or "Choice")
        if simple:
            allowed = tuple(deepcopy(template) for template in templates)
            for template in initial:
                instance = deepcopy(template)
                instance._flow_alternatives = allowed
                instance.is_interchangeable = len(allowed) > 1
                group.add_step(instance)
                instance.is_interchangeable = len(allowed) > 1
        else:
            if node._allow_absence:
                group.also_explore_without = True
            for template in initial:
                group.add_step(template)
                # A whole strategy must never become a global-tag mutation.
                template.is_interchangeable = False
    else:
        group = MetaStep(name=node.alias or "Adaptive group") if isinstance(node, AdaptiveSpec) else MetaOrderedStep(name=node.alias or "Sequence")
        for child in node._children():
            # Adaptive groups deliberately execute only applicable methods.
            child_required = required and not isinstance(node, AdaptiveSpec)
            group.add_step(_compile(child, optimizer, preprocessor, fast,
                                    required=child_required))
    group._flow_node_id = node.node_id
    group._flow_alias = node.alias
    group._flow_explicit = True
    group.tag = node.tag or node.alias
    return group


def compile_pipeline(spec, *, optimizer=GeneticOptimizer, preprocessor=False, fast=False):
    """Validate and compile an isolated snapshot to a fresh runtime Step tree."""
    if not isinstance(spec, (ComponentSpec, GroupSpec)) or spec.kind != "pipeline":
        raise TypeError("pipeline must be a training recipe")
    snapshot = deepcopy(spec)
    if isinstance(snapshot, GroupSpec):
        snapshot.freeze()
    # A preset can operate without minimal predictors in a reduced installation.
    if getattr(snapshot, "_is_default_preset", False):
        empty = []
        for child in snapshot.children:
            if child.alias != "minimal":
                continue
            try:
                predictors = child["minimal_predictor"]
            except KeyError:
                continue
            if (isinstance(predictors, GroupSpec) and predictors.tag == "minimal_predictor"
                    and not predictors._children() and not predictors._registry()):
                empty.append(child)
        for child in empty:
            snapshot.children.remove(child)
    snapshot._check_aliases()
    shapes = _path_shapes(snapshot)
    if shapes != {(1, True)}:
        raise RecipeValidationError("Every candidate path must contain exactly one terminal predictor")
    root = _compile(snapshot, optimizer, preprocessor, fast)
    root._flow_resolved_spec = snapshot
    root._flow_catalogue = tuple(sorted({f"{node.component.__module__}.{node.component.__qualname__}"
                                        for node in snapshot._walk() if isinstance(node, ComponentSpec)}))
    return root
