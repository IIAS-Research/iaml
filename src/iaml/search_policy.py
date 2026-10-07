"""Internal search contracts shared by compiled recipes and optimizers."""
from copy import deepcopy
from hashlib import sha256
from math import isfinite
import pickle
import numpy as np


def parameter_keys(step, ignored=()):
    """Return mutable parameters, honoring domains explicitly declared in recipes."""
    policy = getattr(step, "_flow_parameters", {})
    return [key for key in getattr(step, "configuration", {})
            if (key not in ignored or policy.get(key, {}).get("explicit_domain", False))
            and policy.get(key, {}).get("optimizable", getattr(step, "optimizable", False))
            and not policy.get(key, {}).get("fixed", False)]


def has_finite_range(config):
    """Whether a numeric parameter can be sampled uniformly over its whole range."""
    bounds = config.get("range")
    return bounds is not None and all(bound is not None and isfinite(bound)
                                      for bound in bounds)


def within_parameter_range(config, value):
    """Validate numeric proposals against only the declared, present bounds."""
    if not isfinite(value):
        return False
    if "range" not in config:
        return True
    low, high = config["range"]
    return (low is None or low <= value) and (high is None or value <= high)


def _parameter_signature(step, include_values=False):
    policy = getattr(step, "_flow_parameters", {})
    mutable = set(parameter_keys(step))
    result = []
    for key, config in sorted(step.configuration.items()):
        fixed = key not in mutable
        domain = policy.get(key, {}).get("domain", config.get("range"))
        value = config.get("value") if include_values or fixed else None
        if isinstance(value, np.generic):
            value = value.item()
        result.append((key, fixed, domain, config.get("categorical"),
                       value, bool(policy.get(key, {}).get("explicit_domain", False))))
    return tuple(result)


def policy_signature(step, *, include_alternative_values=True):
    """A semantic search signature; aliases and transient node ids never enter it."""
    if not getattr(step, "_flow_explicit", False):
        return None
    cls = type(step)
    alternatives = tuple(
        (type(item).__module__, type(item).__qualname__,
         bool(item.optimizable),
         bool(getattr(item, '_flow_required', False)),
         _parameter_signature(item, include_values=include_alternative_values))
        for item in getattr(step, "_flow_alternatives", ())
    )
    return (cls.__module__, cls.__qualname__, getattr(step, '_flow_variant_key', None),
            bool(step.optimizable),
            bool(getattr(step, '_flow_required', False)),
            bool(step.is_interchangeable), _parameter_signature(step), alternatives)


def policy_fingerprint(step):
    signature = policy_signature(step)
    if signature is None:
        return None
    return sha256(pickle.dumps(signature, protocol=5)).hexdigest()


def structure_signature(step, ignored=()):
    """Identify a Bayesian space without including its varying observations."""
    cls = type(step)
    mutable = set(parameter_keys(step, ignored))
    parameters = []
    for key, config in sorted(getattr(step, "configuration", {}).items()):
        value = config.get("value")
        if isinstance(value, np.generic):
            value = value.item()
        parameters.append((key, config.get("range"), config.get("categorical"),
                           type(value).__name__, value if key not in mutable else None))
    parameters = tuple(parameters)
    return (cls.__module__, cls.__qualname__, parameters,
            getattr(step, '_flow_variant_key', None),
            policy_signature(step, include_alternative_values=False))


def restore_flow_metadata(source, target):
    """Keep the current recipe's provenance when reusing a cached fitted step."""
    for key in tuple(vars(target)):
        if key.startswith("_flow_"):
            delattr(target, key)
    metadata = deepcopy({key: value for key, value in vars(source).items()
                         if key.startswith("_flow_")})
    for key, value in metadata.items():
        setattr(target, key, value)
    if getattr(source, "_flow_explicit", False):
        target.is_interchangeable = source.is_interchangeable
        target.optimizable = source.optimizable
        target.parents_steps = list(source.parents_steps)
        target._cache_id = source._cache_id
    return target
