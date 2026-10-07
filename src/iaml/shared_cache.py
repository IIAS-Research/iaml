"""Process-shared bounded cache services and their manager."""

from copy import deepcopy
from typing import Any

import multiprocess.managers


class CacheService:
    """Cache partagé, LRU bornée, clé=(fingerprint, df_hash)."""
    def __init__(self, max_cache_size: int = 100) -> None:
        self._saved = None
        self._lru = None
        self._max = max_cache_size
        self._disabled = None
        self._lock = None

    # Keep the shared proxies together in the existing backend initialization API.
    def __set_backend__(self, saved, lru, disabled, lock, maxsize: int):  # pylint: disable=too-many-positional-arguments
        """Attach the shared storage, eviction order, flag, and lock."""
        self._saved = saved
        self._lru = lru
        self._disabled = disabled
        self._lock = lock
        self._max = maxsize

    # API
    def disable(self) -> None:
        """Disable cache reads and writes across workers."""
        with self._lock:
            self._disabled.value = True

    def enable(self) -> None:
        """Enable cache reads and writes across workers."""
        with self._lock:
            self._disabled.value = False

    def get(self, fingerprint: str, df_hash: str) -> Any | None:
        """Return a copy of the cached output and mark it as recently used."""
        if self._disabled.value:
            return None
        key = (fingerprint, df_hash)

        with self._lock:
            if key in self._saved:
                try:
                    self._lru.remove(key)
                except ValueError:
                    pass
                self._lru.append(key)
                return deepcopy(self._saved[key])
        return None

    def put(self, fingerprint: str, df_hash: str, output: Any) -> None:
        """Store a copy of the output and evict the least recently used entries."""
        if self._disabled.value:
            return
        key = (fingerprint, df_hash)

        with self._lock:
            self._saved[key] = deepcopy(output)
            try:
                self._lru.remove(key)
            except ValueError:
                pass
            self._lru.append(key)
            # Éviction LRU
            while len(self._lru) > self._max:
                old_key = self._lru.pop(0)
                self._saved.pop(old_key, None)


# Cache access methods are registered dynamically by start_cache_manager.
class CacheManager(multiprocess.managers.BaseManager):  # pylint: disable=too-few-public-methods
    """Manager exposing a dynamically registered shared cache service."""


def start_cache_manager(max_cache_size: int = 100) -> tuple[CacheManager, CacheService]:
    """
    Démarre un process manager et retourne (manager, cache_proxy).
    À appeler UNE FOIS dans le process parent AVANT de lancer les workers.
    """
    def _cache_factory():
        sm = multiprocess.managers.SyncManager()
        sm.start()
        saved = sm.dict()
        lru = sm.list()
        disabled = sm.Value('b', False)
        lock = sm.RLock()
        cache = CacheService(max_cache_size)
        cache.__set_backend__(saved, lru, disabled, lock, max_cache_size)
        return cache

    CacheManager.register('Cache', callable=_cache_factory)
    mgr = CacheManager()
    mgr.start()
    cache: CacheService = mgr.Cache()
    return mgr, cache
