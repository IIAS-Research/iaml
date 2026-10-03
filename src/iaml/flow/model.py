"""Editable recipes, independent of fitted steps and candidates."""
from __future__ import annotations

from copy import deepcopy
import importlib
import inspect
from numbers import Integral
import uuid

from ..step import Step
from ..metric import Metric
from ..statistic import Statistic
from ..metric_plot import MetricPlot
from .parameters import Const, Int, Float, ParameterSpec


def component_kind(component):
    if not isinstance(component, type):
        raise TypeError("use expects an IAML component class")
    if issubclass(component, Step):
        return "pipeline"
    if issubclass(component, Metric):
        return "metrics"
    if issubclass(component, Statistic):
        return "statistics"
    if issubclass(component, MetricPlot) or getattr(component, "_flow_kind", None) == "explanations":
        return "explanations"
    raise TypeError(f"{component.__name__} is not an IAML training or analysis component")


def as_recipe(value):
    return value if isinstance(value, Recipe) else ComponentSpec(value)


def _backend_parameter_constraints(component):
    """Resolve explicit adapter constraints before the single-backend fallback.

    Composite adapters and custom components can declare
    ``_flow_parameter_constraints`` using sklearn's constraint format. This
    avoids guessing which imported estimator owns each adapter parameter.
    """
    from sklearn.base import BaseEstimator
    explicit = getattr(component, "_flow_parameter_constraints", None)
    if explicit is not None:
        return explicit
    module = inspect.getmodule(component)
    if module is None:
        return {}
    backends = {cls for cls in vars(module).values() if isinstance(cls, type)
                and issubclass(cls, BaseEstimator) and not issubclass(cls, Step)
                and hasattr(cls, "_parameter_constraints")}
    if len(backends) == 1:
        return next(iter(backends))._parameter_constraints
    return {}


def _validate_backend_constraints(component, values, constraints):
    """Validate intrinsic constraints independently of inactive search domains."""
    from sklearn.utils._param_validation import validate_parameter_constraints
    validate_parameter_constraints(constraints, values, component.__name__)


class Recipe:
    """Common navigation, composition and inspection operations."""
    kind = "pipeline"

    def __init__(self):
        self.alias = None
        self.node_id = uuid.uuid4().hex
        self._parent = None
        self._owner = None
        self._owner_field = None

    def __deepcopy__(self, memo):
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if key in ("_parent", "_owner", "_owner_field"):
                setattr(result, key, None)
            else:
                setattr(result, key, deepcopy(value, memo))
        for child in result._children():
            child._parent = result
        return result

    def _children(self):
        return []

    def _walk(self):
        yield self
        for child in self._children():
            yield from child._walk()

    def _root(self):
        node = self
        while node._parent is not None:
            node = node._parent
        return node


    def clone(self):
        result = deepcopy(self)
        def renew(node):
            old = node.node_id
            node.node_id = uuid.uuid4().hex
            mapping = {old: node.node_id}
            if isinstance(node, GroupSpec):
                children = list(node.children) + list(node._materialized.values())
                for child in dict.fromkeys(children):
                    mapping.update(renew(child))
                node._removed_ids = {mapping.get(identity, identity) for identity in node._removed_ids}
            return mapping
        renew(result)
        return result

    def _check_aliases(self):
        aliases = set()
        for node in self._walk():
            if node.alias is not None:
                if node.alias in aliases:
                    raise ValueError(f"Duplicate alias {node.alias!r}")
                aliases.add(node.alias)

    def named(self, alias):
        if not isinstance(alias, str) or not alias:
            raise ValueError("An alias must be a nonempty string")
        if any(node is not self and node.alias == alias for node in self._root()._walk()):
            raise ValueError(f"Duplicate alias {alias!r}")
        previous = self.alias
        if isinstance(self._parent, ChoiceSpec) and self._parent.initial_aliases is not None:
            self._parent.initial_aliases = tuple(alias if value == previous else value
                                                 for value in self._parent.initial_aliases)
        self.alias = alias
        return self



    def __iter__(self):
        return iter(self._children())

    def __rshift__(self, other):
        other = as_recipe(other)
        if self.kind != "pipeline" or other.kind != "pipeline":
            raise TypeError("Only training recipes can be composed with >>")
        children = []
        for part in (self, other):
            if isinstance(part, SequenceSpec) and part.alias is None:
                children.extend(part._children())
            else:
                children.append(part)
        return SequenceSpec(children)







class ComponentSpec(Recipe):
    """A configured class recipe; constructing it never fits the component."""

    def __init__(self, component, **params):
        super().__init__()
        self.component = component
        self.kind = component_kind(component)
        self.parameters = {}
        self._declared_params = set()
        self._constructor_defaults = {}
        if self.kind == "pipeline":
            try:
                instance = component()
            except TypeError as error:
                raise TypeError(f"{component.__name__} must support an unfitted default constructor") from error
            for key, conf in instance.configuration.items():
                self._constructor_defaults[key] = deepcopy(conf.get("default", conf.get("value")))
                self.parameters[key] = ParameterSpec(
                    deepcopy(conf.get("value", conf.get("default"))),
                    tuple(conf["range"]) if "range" in conf else None,
                    categorical=tuple(conf["categorical"]) if "categorical" in conf else None,
                    optimizable=instance.optimizable,
                )
        else:
            signature = inspect.signature(component.__init__)
            for key, parameter in signature.parameters.items():
                if key != "self" and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL):
                    if parameter.default is not parameter.empty:
                        self._constructor_defaults[key] = deepcopy(parameter.default)
                        self.parameters[key] = ParameterSpec(deepcopy(parameter.default), fixed=True)
        self.configure(**params)

    def _prepare(self, params):
        prepared = deepcopy(self.parameters)
        if not params:
            return prepared
        signature = inspect.signature(self.component.__init__)
        constraints = _backend_parameter_constraints(self.component) if self.kind == "pipeline" else {}
        for key, value in params.items():
            if self.kind == "pipeline":
                if key not in prepared:
                    raise AttributeError(f"Unknown parameter {key!r} for {self.component.__name__}")
                prepared[key] = prepared[key].updated(value)
                default = self._constructor_defaults[key]
                actual = prepared[key].value
                if key not in constraints and isinstance(default, bool) and not isinstance(actual, bool):
                    raise TypeError(f"{self.component.__name__}.{key} requires a boolean")
                if isinstance(default, Integral) and not isinstance(default, bool) and (
                    isinstance(actual, bool) or (key not in constraints and not isinstance(actual, Integral))
                ):
                    raise TypeError(f"{self.component.__name__}.{key} requires an integer")
            else:
                if isinstance(value, (Int, Float)):
                    raise TypeError(f"Analysis parameter {key!r} is fixed; search domains are unsupported")
                if key not in signature.parameters or key == "self":
                    raise AttributeError(f"Unknown parameter {key!r} for {self.component.__name__}")
                actual = value.value if isinstance(value, Const) else value
                prepared[key] = ParameterSpec(deepcopy(actual), fixed=True)
        if self.kind != "pipeline":
            kwargs = {key: param.value for key, param in prepared.items()}
            try:
                self.component(**kwargs)
            except Exception as error:
                raise ValueError(f"Invalid configuration of {self.component.__name__}: {error}") from error
        else:
            _validate_backend_constraints(self.component, {key: prepared[key].value for key in params}, constraints)
        return prepared

    def configure(self, **params):
        prepared = self._prepare(params)
        self.parameters = prepared
        self._declared_params.update(params)
        return self

    def instantiate(self):
        """Create a fresh unfitted component carrying its declared policy."""
        if self.kind != "pipeline":
            return self.component(**{key: deepcopy(param.value) for key, param in self.parameters.items()})
        instance = self.component()
        policy = {}
        for key, param in self.parameters.items():
            conf = instance.configuration[key]
            conf["value"] = deepcopy(param.value)
            if param.fixed:
                conf.pop("range", None)
                conf.pop("categorical", None)
            elif param.explicit_domain:
                conf.pop("categorical", None)
                conf["range"] = list(param.domain)
            policy[key] = {"fixed": param.fixed, "domain": list(param.domain) if param.domain else None,
                           "optimizable": param.optimizable and not param.fixed,
                           "explicit_domain": param.explicit_domain}
        instance._flow_parameters = policy
        instance._flow_node_id = self.node_id
        instance._flow_variant_id = self.node_id
        instance._flow_alias = self.alias
        instance._flow_explicit = True
        instance.is_interchangeable = False
        if any(p.optimizable and not p.fixed for p in self.parameters.values()):
            instance.optimizable = True
        return instance




class GroupSpec(Recipe):
    mode = "sequence"

    def __init__(self, children=(), *, tag=None):
        super().__init__()
        self.children = []
        self.tag = tag
        self.excluded = set()
        self._materialized = {}
        self._replaced_slots = set()
        self._removed_ids = set()
        self._frozen = False
        for child in children:
            self.children.append(self._coerce(child).clone())
        for child in self.children:
            child._parent = self
        self._check_aliases()

    def _coerce(self, child):
        recipe = as_recipe(child)
        if recipe.kind != self.kind:
            raise TypeError(f"Expected {self.kind} recipe, received {recipe.kind}")
        return recipe


    def _children(self):
        return list(self.children)








class SequenceSpec(GroupSpec):
    mode = "sequence"


class AdaptiveSpec(GroupSpec):
    mode = "adaptive"


class ChoiceSpec(GroupSpec):
    mode = "choice"

    def __init__(self, children=(), *, tag=None):
        self.initial_aliases = None
        self._preset_start = None
        self._allow_absence = False
        super().__init__(children, tag=tag)



class PipelineSpec(ChoiceSpec):
    """Root of the default recipe, containing main and minimal strategies."""




def use(component, **params):
    return ComponentSpec(component, **params)


def choice(*alternatives):
    return ChoiceSpec(alternatives)
