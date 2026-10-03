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
