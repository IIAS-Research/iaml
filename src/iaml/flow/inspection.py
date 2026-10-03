"""Structured inspection and executable declarative reconstruction."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from difflib import unified_diff
import json

from .model import (AdaptiveSpec, AnalysisCollection, ChoiceSpec, ComponentSpec,
                    GroupSpec, PipelineSpec, SequenceSpec)


def _class_name(component):
    return f"{component.__module__}.{component.__qualname__}"


def _parameter_state(parameter):
    return {"value": deepcopy(parameter.value), "domain": deepcopy(parameter.domain),
            "fixed": parameter.fixed, "optimizable": parameter.optimizable and not parameter.fixed,
            "categorical": deepcopy(parameter.categorical), "explicit_domain": parameter.explicit_domain}


def _state(recipe):
    payload = {"alias": recipe.alias, "kind": recipe.kind}
    if isinstance(recipe, ComponentSpec):
        payload.update(component=_class_name(recipe.component),
                       parameters={key: _parameter_state(value) for key, value in recipe.parameters.items()})
        if getattr(recipe, "_conditional_missing", False):
            payload["condition"] = "missing values in the generation input"
    else:
        payload.update(mode=recipe.mode, tag=recipe.tag, resolved=recipe._frozen,
                       excluded=sorted(_class_name(cls) for cls in recipe.excluded),
                       children=[_state(child) for child in recipe._children()])
        if isinstance(recipe, ChoiceSpec):
            payload.update(initial=recipe.initial_aliases, optional=recipe._allow_absence,
                           preset_start=recipe._preset_start)
    return payload


@dataclass(frozen=True)
class RecipeReport(Mapping):
    """A structured report which also prints as a readable tree."""
    data: dict

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def to_dict(self):
        from copy import deepcopy
        return deepcopy(self.data)

    def __str__(self):
        if "changes" in self.data:
            return self.data["changes"] or "No declarative changes"
        if "tree" not in self.data:
            return json.dumps(self.data, indent=2, default=str, ensure_ascii=False)
        lines = []
        def render(node, level=0):
            label = node.get("alias") or node.get("component", node.get("mode", "recipe"))
            suffix = ""
            if node.get("tag"):
                suffix = f" [family {node['tag']}, {'resolved' if node['resolved'] else 'open'}]"
            if node.get("initial") is not None:
                suffix += f" [start={node['initial']}]"
            lines.append("  " * level + str(label) + suffix)
            for key, value in node.get("parameters", {}).items():
                mode = "fixed" if value["fixed"] else "initial"
                domain = value["domain"] or value["categorical"]
                retained = "retained domain" if value["fixed"] else "domain"
                lines.append("  " * (level + 1) + f"{key}: {mode}={value['value']!r}"
                             + (f", {retained}={domain!r}" if domain is not None else ""))
            for child in node.get("children", []):
                render(child, level + 1)
        render(self.data["tree"])
        return "\n".join(lines)


def describe(recipe):
    return RecipeReport({"tree": _state(recipe)})


def diff(recipe, base):
    before = json.dumps(_state(base), indent=2, sort_keys=True, default=repr).splitlines()
    after = json.dumps(_state(recipe), indent=2, sort_keys=True, default=repr).splitlines()
    return RecipeReport({"changes": "\n".join(unified_diff(before, after, fromfile="base", tofile="recipe")),
                         "before": _state(base), "after": _state(recipe)})


class _CodeBuilder:
    def __init__(self):
        self.lines = []
        self.imports = {"from iaml.flow import PipelineSpec, Const, Int, Float, use, choice, optional, metrics, statistics, explanations"}
        self.classes = {}
        self.index = 0
        self.aliases = set()

    def component(self, cls):
        if cls not in self.classes:
            if "<locals>" in cls.__qualname__:
                raise ValueError("to_code requires importable component classes")
            alias = f"_component{len(self.classes)}"
            first, *rest = cls.__qualname__.split(".")
            self.imports.add(f"from {cls.__module__} import {first} as {alias}")
            self.classes[cls] = alias + ("." + ".".join(rest) if rest else "")
        return self.classes[cls]

    def literal(self, value):
        if callable(value):
            module, name = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
            if not module or not name or "<locals>" in name or name == "<lambda>":
                raise ValueError("to_code requires importable callable parameter values")
            alias = f"_value{len(self.imports)}"
            first, *rest = name.split(".")
            self.imports.add(f"from {module} import {first} as {alias}")
            return alias + ("." + ".".join(rest) if rest else "")
        if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict, set)):
            return repr(value)
        raise TypeError(f"to_code cannot represent parameter value {type(value).__name__}")

    def configuration(self, target, node):
        for key in sorted(node._declared_params):
            parameter = node.parameters[key]
            value = self.literal(parameter.value)
            if parameter.explicit_domain:
                low, high = parameter.domain
                domain_type = "Int" if all(isinstance(v, int) and not isinstance(v, bool)
                                           for v in (low, high)) else "Float"
                initial = parameter.value
                if parameter.fixed:
                    initial = parameter._retained_initial if parameter._retained_initial is not None else low
                declaration = f"{domain_type}({low!r}, {high!r}, initial={initial!r})"
                self.lines.append(f"{target}.configure({key}={declaration})")
                if parameter.fixed:
                    self.lines.append(f"{target}.configure({key}=Const({value}))")
            else:
                self.lines.append(f"{target}.configure({key}={'Const(' + value + ')' if parameter.fixed and node.kind == 'pipeline' else value})")

    def node(self, node):
        variable = f"_recipe{self.index}"
        self.index += 1
        if isinstance(node, ComponentSpec):
            if getattr(node, "_conditional_missing", False):
                self.lines.append(f"{variable} = PipelineSpec.default().minimal_imputer.clone()")
            else:
                # Required analytic constructor arguments need to be present in
                # the initial use(), rather than configured after construction.
                params = ""
                if node.kind != "pipeline":
                    params = "".join(f", {key}={self.literal(param.value)}" for key, param in node.parameters.items())
                self.lines.append(f"{variable} = use({self.component(node.component)}{params})")
            self.configuration(variable, node)
        elif isinstance(node, AnalysisCollection):
            if node.tag is not None and not node._frozen:
                self.lines.append(f"{variable} = {node.kind}()")
                self.family(variable, node)
            else:
                children = [self.node(child) for child in node._children()]
                if children:
                    self.lines.append(f"{variable} = {node.kind}({', '.join(children)})")
                else:
                    self.imports.add("from iaml.flow import AnalysisCollection")
                    self.lines.append(f"{variable} = AnalysisCollection({node.kind!r}, [])")
        elif isinstance(node, GroupSpec) and node.tag is not None and not node._frozen:
            if isinstance(node, AdaptiveSpec):
                self.imports.add("from iaml.flow.model import AdaptiveSpec")
                self.lines.append(f"{variable} = AdaptiveSpec(tag={node.tag!r})")
            else:
                self.lines.append(f"{variable} = choice(tag={node.tag!r})")
                self.lines.append(f"{variable}._allow_absence = {node._allow_absence!r}")
                self.lines.append(f"{variable}._preset_start = {node._preset_start!r}")
            self.family(variable, node)
        elif isinstance(node, PipelineSpec):
            children = [self.node(child) for child in node._children()]
            self.lines.append(f"{variable} = PipelineSpec.default()")
            self.lines.append(f"{variable}.remove('minimal', 'main')")
            for child in children:
                self.lines.append(f"{variable}.add({child})")
        elif isinstance(node, ChoiceSpec):
            children = [self.node(child) for child in node._children()]
            constructor = f"choice({', '.join(children)})"
            if node._allow_absence:
                if not children:
                    raise ValueError("to_code requires optional to contain at least one recipe")
                constructor = f"optional({children[0]})"
            self.lines.append(f"{variable} = {constructor}")
            if node._allow_absence:
                for child in children[1:]:
                    self.lines.append(f"{variable}.add({child})")
        elif isinstance(node, AdaptiveSpec):
            children = [self.node(child) for child in node._children()]
            self.imports.add("from iaml.flow.model import AdaptiveSpec")
            self.lines.append(f"{variable} = AdaptiveSpec([{', '.join(children)}])")
        else:
            children = [self.node(child) for child in node._children()]
            # Using >> here would collapse single-child sequences and flatten
            # unnamed nested sequences, losing their editable containers.
            self.imports.add("from iaml.flow.model import SequenceSpec")
            self.lines.append(f"{variable} = SequenceSpec([{', '.join(children)}])")
        if node.alias is not None:
            self.lines.append(f"{variable}.named({node.alias!r})")
        if isinstance(node, GroupSpec) and node._frozen:
            # A resolved snapshot reconstructs its exact catalogue, rather than
            # expanding an open family against a different environment.
            self.lines.append(f"{variable}._frozen = True")
            if node.tag is not None:
                self.lines.append(f"{variable}.tag = {node.tag!r}")
            if node.excluded:
                exclusions = ", ".join(self.component(cls) for cls in sorted(node.excluded, key=_class_name))
                self.lines.append(f"{variable}.excluded = {{{exclusions}}}")
            if isinstance(node, ChoiceSpec):
                self.lines.append(f"{variable}._preset_start = {node._preset_start!r}")
        if isinstance(node, ChoiceSpec) and node.initial_aliases is not None:
            self.lines.append(f"{variable}.start({', '.join(repr(alias) for alias in node.initial_aliases)})")
        return variable

    def family(self, variable, node):
        # Registry variants can carry configurations/aliases independently of
        # explicitly added variants of the same class.
        for component, child in node._materialized.items():
            if component in node.excluded:
                continue
            if component in node._replaced_slots and child.node_id not in node._removed_ids:
                continue
            target = f"next(iter({variable}.find_all({self.component(component)})))"
            if child.node_id in node._removed_ids:
                # Class removal would add an exclusion and change the family's
                # declaration. Give an unnamed removed slot a temporary alias.
                index = 0
                while f"_iaml_removed_{index}" in self.aliases:
                    index += 1
                alias = f"_iaml_removed_{index}"
                self.aliases.add(alias)
                self.lines.append(f"{target}.named({alias!r})")
                self.lines.append(f"{variable}.remove({alias!r})")
            elif isinstance(child, ComponentSpec) and child.component is component:
                if not child._declared_params and child.alias is None:
                    continue
                self.configuration(target, child)
                if child.alias is not None:
                    self.lines.append(f"{target}.named({child.alias!r})")
        for component in sorted(node.excluded, key=_class_name):
            self.lines.append(f"{variable}.remove({self.component(component)})")
        # Restore replacements after automatic exclusions, so excluding their
        # class does not remove an explicitly authorized replacement variant.
        for component, child in node._materialized.items():
            if component not in node._replaced_slots or child.node_id in node._removed_ids:
                continue
            child_variable = self.node(child)
            self.lines.append(f"{variable}._restore_registry_replacement({self.component(component)}, {child_variable})")
        for child in node.children:
            child_variable = self.node(child)
            self.lines.append(f"{variable}.add({child_variable})")


def to_code(recipe):
    builder = _CodeBuilder()
    builder.aliases = {node.alias for node in recipe._walk() if node.alias is not None}
    variable = builder.node(recipe)
    name = "pipeline" if recipe.kind == "pipeline" else "analyses"
    return "\n".join([*sorted(builder.imports), "", *builder.lines, f"{name} = {variable}", ""])
