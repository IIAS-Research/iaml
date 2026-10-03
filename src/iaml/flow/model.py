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

    def attach(self, owner, field):
        """Attach a root to its study; used by the study integration layer."""
        self._owner, self._owner_field = owner, field
        return self

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

    def __getitem__(self, alias):
        if not isinstance(alias, str):
            raise TypeError("Recipe selectors must be aliases; use find_all for a class")
        for node in self._walk():
            if node.alias == alias:
                return node
        raise KeyError(f"Unknown alias {alias!r}")

    def __getattr__(self, alias):
        if alias.startswith("_"):
            raise AttributeError(alias)
        try:
            return self[alias]
        except KeyError as error:
            raise AttributeError(alias) from error

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

    def find_all(self, component):
        component_kind(component)
        return Selection(self, [node for node in self._walk()
                                if isinstance(node, ComponentSpec) and node.component is component])

    def replace(self, recipe):
        if self._parent is None and self._owner is None:
            raise ValueError("replace requires an attached recipe")
        replacement = as_recipe(recipe).clone()
        if replacement.kind != self.kind:
            raise TypeError("Replacement belongs to a different component family")
        if self._parent is not None:
            # Apply the same container contract as construction and add before
            # changing aliases, ownership, or the parent's contents.
            self._parent._coerce(replacement)
        if replacement.alias not in (None, self.alias):
            raise ValueError("Replacement root must have no alias or the existing alias")
        replacement.alias = self.alias
        remaining = {n.alias for n in self._root()._walk() if n not in list(self._walk())
                     and n.alias is not None}
        replacement._check_aliases()
        if remaining.intersection(n.alias for n in replacement._walk() if n.alias is not None):
            raise ValueError("Replacement introduces a duplicate alias")
        if self._parent is not None:
            parent = self._parent
            parent._replace_child(self, replacement)
            replacement._parent = parent
        else:
            owner, field = self._owner, self._owner_field
            replacement.attach(owner, field)
            setattr(owner, field, replacement)
        self._parent = self._owner = self._owner_field = None
        return replacement

    def describe(self):
        from .inspection import describe
        return describe(self)

    def diff(self, base):
        from .inspection import diff
        return diff(self, base)

    def to_code(self):
        from .inspection import to_code
        return to_code(self)


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


class Selection:
    """A snapshot of component objects, with atomic collective configuration."""
    def __init__(self, scope, targets):
        self.scope, self.targets = scope, tuple(targets)
        self._root = scope._root()
        self._owner = self._root._owner
        self._owner_field = self._root._owner_field

    def __iter__(self):
        return iter(self.targets)

    def __len__(self):
        return len(self.targets)

    def configure(self, **params):
        if not self.targets:
            return self
        if self.scope._root() is not self._root or (
            self._owner is not None and getattr(self._owner, self._owner_field, None) is not self._root
        ):
            raise ValueError("Selection scope was detached; call find_all again")
        current = set(self.scope._walk())
        if any(target not in current for target in self.targets):
            raise ValueError("Selection contains detached components; call find_all again")
        prepared = [(target, target._prepare(params)) for target in self.targets]
        for target, configuration in prepared:
            target.parameters = configuration
            target._declared_params.update(params)
        return self


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

    def _registry(self):
        return [cls for cls, tags in Step.available_steps.items() if self.tag in tags]

    def _children(self):
        if self.tag is None or self._frozen:
            return list(self.children)
        found = []
        catalogue = set(self._registry()).union(self._replaced_slots)
        for component in sorted(catalogue, key=lambda cls: (cls.__module__, cls.__qualname__)):
            if component in self.excluded and component not in self._replaced_slots:
                continue
            if component not in self._materialized:
                node = ComponentSpec(component)
                node._parent = self
                self._materialized[component] = node
            node = self._materialized[component]
            if node.node_id not in self._removed_ids:
                found.append(node)
        return found + list(self.children)

    def _replace_child(self, old, new):
        if old in self.children:
            self.children[self.children.index(old)] = new
        else:
            # Registry entries have stable slots too. Retain the registry key
            # even when a replacement uses a different class or group mode.
            component = next(cls for cls, child in self._materialized.items() if child is old)
            self._materialized[component] = new
            self._replaced_slots.add(component)

    def _restore_registry_replacement(self, component, recipe):
        """Reconstruct an explicit replacement at its original registry slot.

        The original component can be excluded or absent from the current
        registry, while the explicitly supplied replacement remains allowed.
        """
        node = self._coerce(recipe).clone()
        node._check_aliases()
        old = self._materialized.get(component)
        replaced = set(old._walk()) if old is not None else set()
        aliases = {n.alias for n in self._root()._walk() if n not in replaced and n.alias is not None}
        if aliases.intersection(n.alias for n in node._walk() if n.alias is not None):
            raise ValueError("Replacement introduces a duplicate alias")
        if old is not None:
            old._parent = None
        self._materialized[component] = node
        self._replaced_slots.add(component)
        node._parent = self
        return self

    def add(self, recipe, before=None):
        node = self._coerce(recipe).clone()
        if before is not None:
            anchor = self[before]
            parent = anchor._parent
            if not isinstance(parent, SequenceSpec):
                raise ValueError("before must designate an item of a sequence")
            return self._insert(node, parent, parent.children.index(anchor))
        return self._insert(node, self, len(self.children))

    def _insert(self, node, parent, index):
        node._check_aliases()
        aliases = {n.alias for n in self._root()._walk() if n.alias is not None}
        if aliases.intersection(n.alias for n in node._walk() if n.alias is not None):
            raise ValueError("Addition introduces a duplicate alias")
        parent.children.insert(index, node)
        node._parent = parent
        # An explicit variant is authorized through children even if its class
        # remains excluded from automatic registry expansion. This must not
        # resurrect the default variant that the user removed.
        return self

    def remove(self, *selectors):
        targets = []
        for selector in selectors:
            if isinstance(selector, str):
                node = self[selector]
                if node is self:
                    raise ValueError("A group cannot remove itself")
                targets.append(node)
            elif isinstance(selector, type):
                matches = [node for node in self._children()
                           if isinstance(node, ComponentSpec) and node.component is selector]
                if not matches:
                    raise ValueError(f"Component {selector.__name__} is absent from this group")
                targets.extend(matches)
            else:
                raise TypeError("remove expects aliases or component classes")
        for node in targets:
            parent = node._parent
            if isinstance(parent, ChoiceSpec) and parent.initial_aliases is not None and node.alias in parent.initial_aliases:
                raise ValueError(f"Call start() or choose another start before removing {node.alias!r}")
        for node in dict.fromkeys(targets):
            parent = node._parent
            if node in parent.children:
                parent.children.remove(node)
            else:
                parent._removed_ids.add(node.node_id)
            if parent.tag is not None and isinstance(node, ComponentSpec):
                # Class removal excludes every registered occurrence, while alias
                # removal keeps other variants of the class available.
                if node.component in selectors:
                    parent.excluded.add(node.component)
            node._parent = None
        return self

    def freeze(self):
        children = self._children()
        self.children = children
        self._materialized = {}
        self._replaced_slots = set()
        self._frozen = True
        for child in children:
            child._parent = self
            if isinstance(child, GroupSpec):
                child.freeze()
        return self


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

    def start(self, *aliases):
        if len(set(aliases)) != len(aliases):
            raise ValueError("start aliases must be distinct")
        available = {node.alias for node in self._children() if node.alias is not None}
        if any(alias not in available for alias in aliases):
            raise KeyError("start requires aliases of direct alternatives")
        self.initial_aliases = tuple(aliases) if aliases else None
        self._preset_start = None
        return self


class PipelineSpec(ChoiceSpec):
    """Root of the default recipe, containing main and minimal strategies."""
    @classmethod
    def default(cls):
        from .compiler import default_spec
        return default_spec()


class AnalysisCollection(GroupSpec):
    """All requested analytic computations, rather than candidate alternatives."""
    mode = "analyses"

    def __init__(self, kind, children=(), *, defaults=False):
        self.kind = kind
        super().__init__(children, tag=kind if defaults else None)

    def _coerce(self, child):
        node = as_recipe(child)
        if not isinstance(node, ComponentSpec) or node.kind != self.kind:
            raise TypeError(f"Expected a {self.kind} component")
        return node

    def _registry(self):
        if self.kind == "explanations":
            module = importlib.import_module("iaml.plots")
            found = {value for value in vars(module).values() if isinstance(value, type)
                     and issubclass(value, MetricPlot) and value is not MetricPlot}
            try:
                found.add(importlib.import_module("iaml.explainers").KernelSHAP)
            except (ImportError, AttributeError):
                pass
            return found
        base = Metric if self.kind == "metrics" else Statistic
        module = importlib.import_module(f"iaml.{self.kind}")
        public = {value for value in vars(module).values() if isinstance(value, type)
                  and issubclass(value, base) and value is not base}
        # Imported experimental IAML components stay out of automatic families.
        public.update(cls for cls in base.all_subclasses() if not cls.__module__.startswith("iaml."))
        return public

    def resolved(self):
        self._check_aliases()
        return list(self._children())


def use(component, **params):
    return ComponentSpec(component, **params)


def choice(*alternatives, tag=None):
    if alternatives and tag is not None:
        raise ValueError("choice accepts alternatives or tag, not both")
    return ChoiceSpec(alternatives, tag=tag)


def normalizers():
    return choice(tag="normalize")


def predictors():
    return choice(tag="predictor")


def optional(recipe):
    result = ChoiceSpec([recipe])
    result._allow_absence = True
    return result


def metrics(*components):
    return AnalysisCollection("metrics", components, defaults=not components)


def statistics(*components):
    return AnalysisCollection("statistics", components, defaults=not components)


def explanations(*components):
    return AnalysisCollection("explanations", components, defaults=not components)
