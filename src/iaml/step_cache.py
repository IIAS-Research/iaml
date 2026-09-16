"""Shared in-process cache for step outputs."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any
from weakref import ReferenceType, ref

from .meta_singleton import MetaSingleton


class StepCache(metaclass=MetaSingleton):
    """LRU cache shared across deep-copied steps, validating input identity."""
    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._data: OrderedDict[tuple, tuple[ReferenceType | None, Any]] = OrderedDict()
        self._by_step: dict[str, set[tuple]] = {}
        self._lock = RLock()

    def get(self, key: tuple, input_candidate: Any) -> Any | None:
        """Return an output only while its original input still matches by identity."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            input_ref, value = entry
            original_input = input_ref() if input_ref is not None else None
            if original_input is not input_candidate or (
                input_ref is not None and original_input is None
            ):
                self._remove(key)
                return None
            if value is None:
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: tuple, value: Any, step_cache_id: str, input_candidate: Any) -> None:
        """Cache an output without keeping its input candidate alive."""
        with self._lock:
            input_ref = ref(input_candidate) if input_candidate is not None else None
            if key in self._data:
                self._data.move_to_end(key)
            else:
                self._by_step.setdefault(step_cache_id, set()).add(key)
            self._data[key] = (input_ref, value)
            self._evict()

    def _remove(self, key: tuple) -> None:
        """Remove an entry and its step index while holding the cache lock."""
        self._data.pop(key, None)
        step_cache_id = key[0]
        keys = self._by_step.get(step_cache_id)
        if keys is not None:
            keys.discard(key)
            if not keys:
                self._by_step.pop(step_cache_id, None)

    def _evict(self) -> None:
        while len(self._data) > self._max_size:
            self._remove(next(iter(self._data)))

    def clear(self, step_cache_id: str) -> None:
        with self._lock:
            keys = self._by_step.pop(step_cache_id, None)
            if not keys:
                return
            for key in keys:
                self._data.pop(key, None)

    def values_for_step(self, step_cache_id: str) -> list[Any]:
        with self._lock:
            if step_cache_id not in self._by_step:
                return []
            return [
                value for key, (_, value) in self._data.items()
                if key[0] == step_cache_id
            ]

    def size_for_step(self, step_cache_id: str) -> int:
        with self._lock:
            keys = self._by_step.get(step_cache_id)
            return len(keys) if keys else 0

    def total_size(self) -> int:
        with self._lock:
            return len(self._data)
