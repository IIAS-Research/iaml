"""Values and search domains used by declarative recipes."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class Const:
    """Keep a parameter fixed without discarding its previous search domain."""
    value: Any


@dataclass(frozen=True)
class Int:
    """An inclusive integer search interval and its starting value."""
    low: int
    high: int
    initial: int

    def __post_init__(self):
        values = (self.low, self.high, self.initial)
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in values):
            raise TypeError("Int requires integer bounds and initial value")
        if not self.low <= self.initial <= self.high:
            raise ValueError("Int requires low <= initial <= high")


@dataclass(frozen=True)
class Float:
    """An inclusive real search interval and its starting value."""
    low: float
    high: float
    initial: float

    def __post_init__(self):
        values = (self.low, self.high, self.initial)
        if any(isinstance(v, bool) or not isinstance(v, Real) for v in values):
            raise TypeError("Float requires real bounds and initial value")
        if not all(isfinite(v) for v in values) or not self.low <= self.initial <= self.high:
            raise ValueError("Float requires finite low <= initial <= high")


@dataclass
class ParameterSpec:
    """Effective value, retained domain and optimization eligibility."""
    value: Any
    domain: tuple | None = None
    fixed: bool = False
    categorical: tuple | None = None
    explicit_domain: bool = False
    optimizable: bool = False
    _retained_initial: Any = field(default=None, repr=False, compare=False)

    def updated(self, value):
        from copy import deepcopy
        result = deepcopy(self)
        if isinstance(value, Const):
            if not self.fixed:
                result._retained_initial = deepcopy(self.value)
            result.value, result.fixed = deepcopy(value.value), True
        elif isinstance(value, (Int, Float)):
            cast = int if isinstance(value, Int) else float
            result.value = cast(value.initial)
            result.domain = (cast(value.low), cast(value.high))
            result.categorical = None
            result.fixed = False
            result.explicit_domain = True
            result.optimizable = True
            result._retained_initial = None
        else:
            result.value, result.fixed = deepcopy(value), False
            result._retained_initial = None
        result.validate()
        return result

    def validate(self):
        if self.fixed:
            return
        if self.domain is not None:
            low, high = self.domain
            try:
                valid = (low is None or low <= self.value) and (high is None or self.value <= high)
            except TypeError as error:
                raise TypeError(f"Value {self.value!r} is incompatible with domain {self.domain}") from error
            if not valid:
                raise ValueError(f"Initial value {self.value!r} is outside {self.domain}")
        if self.categorical is not None and self.value not in self.categorical:
            raise ValueError(f"Initial value {self.value!r} is outside {self.categorical}")
