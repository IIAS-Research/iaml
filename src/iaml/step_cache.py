"""Shared in-process cache for step outputs."""
from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any

from .meta_singleton import MetaSingleton


class StepCache(metaclass=MetaSingleton):
    """LRU cache shared across deep-copied steps."""
    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._data: OrderedDict[tuple, Any] = OrderedDict()
        self._by_step: dict[str, set[tuple]] = {}
        self._lock = RLock()

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: tuple, value: Any, step_cache_id: str) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
            else:
                self._data[key] = value
                self._by_step.setdefault(step_cache_id, set()).add(key)
            self._evict()

    def _evict(self) -> None:
        while len(self._data) > self._max_size:
            old_key, _ = self._data.popitem(last=False)
            step_cache_id = old_key[0]
            keys = self._by_step.get(step_cache_id)
            if keys:
                keys.discard(old_key)
                if not keys:
                    self._by_step.pop(step_cache_id, None)

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
                value for key, value in self._data.items()
                if key[0] == step_cache_id
            ]

    def size_for_step(self, step_cache_id: str) -> int:
        with self._lock:
            keys = self._by_step.get(step_cache_id)
            return len(keys) if keys else 0

    def total_size(self) -> int:
        with self._lock:
            return len(self._data)
