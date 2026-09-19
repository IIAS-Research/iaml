from collections import deque
from copy import deepcopy
from typing import Any, Tuple
import multiprocess.managers
from .logger import Logger


class CacheService:
    """Cache partagé, LRU bornée, clé=(fingerprint, df_hash)."""
    def __init__(self, max_cache_size: int = 100) -> None:
        self._saved = None
        self._lru = None
        self._max = max_cache_size
        self._disabled = None
        self._lock = None

    def __set_backend__(self, saved, lru, disabled, lock, maxsize: int):
        self._saved = saved
        self._lru = lru
        self._disabled = disabled
        self._lock = lock
        self._max = maxsize

    # API
    def disable(self) -> None:
        with self._lock:
            self._disabled.value = True

    def enable(self) -> None:
        with self._lock:
            self._disabled.value = False

    def get(self, fingerprint: str, df_hash: str) -> Any | None:
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


class CacheManager(multiprocess.managers.BaseManager):
    pass

def start_cache_manager(max_cache_size: int = 100) -> tuple[CacheManager, CacheService]:
    """
    Démarre un process manager et retourne (manager, cache_proxy).
    À appeler UNE FOIS dans le process parent AVANT de lancer les workers.
    """
    def _cache_factory():
        from multiprocess.managers import SyncManager
        sm = SyncManager()
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
