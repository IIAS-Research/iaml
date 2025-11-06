# cache_singleton.py
from typing import Any
import pandas as pd

from .meta_singleton import MetaSingleton
from .cache_keys import hash_df
from .logger import Logger

class Cache(metaclass=MetaSingleton):
    """Façade singleton vers un cache partagé."""
    def __init__(self) -> None:
        self._backend = None

    def configure(self, backend) -> None:
        """Brancher le proxy partagé (à faire dans le parent ET dans chaque worker)."""
        self._backend = backend

    def disable(self) -> None:
        if self._backend: self._backend.disable()

    def enable(self) -> None:
        if self._backend: self._backend.enable()

    def from_cache(self, fingerprint: str, dataset: pd.DataFrame) -> Any | None:
        if not self._backend:
            return None
        try:
            return self._backend.get(fingerprint, hash_df(dataset))
        except (EOFError, OSError) as exc:
            Logger().warning(f"Shared cache disabled after get failure: {exc!r}")
            self._backend = None
            return None

    def add_to_cache(self, fingerprint: str, dataset: pd.DataFrame, output: Any) -> None:
        if not self._backend:
            return
        try:
            self._backend.put(fingerprint, hash_df(dataset), output)
        except (EOFError, OSError) as exc:
            Logger().warning(f"Shared cache disabled after put failure: {exc!r}")
            self._backend = None
