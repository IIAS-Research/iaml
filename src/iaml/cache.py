# cache_singleton.py
from typing import Any
import pickle
import pandas as pd

from .meta_singleton import MetaSingleton
from .cache_keys import hash_df
from .dataset import Dataset
from .logger import Logger


def _data_fingerprint(dataset: Dataset | pd.DataFrame | str) -> str:
    """Accept complete datasets, frozen keys, and legacy feature-only inputs."""
    if isinstance(dataset, str):
        return dataset
    if isinstance(dataset, Dataset):
        return dataset.fingerprint()
    return hash_df(dataset)


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

    def from_cache(self, fingerprint: str, dataset: Dataset | pd.DataFrame | str) -> Any | None:
        """Look up an operation using all dataset inputs or a precomputed key.

        DataFrame inputs remain supported for feature-only operations. Supervised
        operations must pass a Dataset, or its fingerprint captured before mutation.
        """
        if not self._backend:
            return None
        try:
            return self._backend.get(fingerprint, _data_fingerprint(dataset))
        except (EOFError, OSError) as exc:
            Logger().warning(f"Shared cache disabled after get failure: {exc!r}")
            self._backend = None
            return None

    def add_to_cache(
        self, fingerprint: str, dataset: Dataset | pd.DataFrame | str, output: Any
    ) -> None:
        """Store an operation result under the same input key used for lookup."""
        if not self._backend:
            return
        try:
            self._backend.put(fingerprint, _data_fingerprint(dataset), output)
        except (EOFError, OSError, pickle.PicklingError, TypeError) as exc:
            Logger().warning(f"Shared cache disabled after put failure: {exc!r}")
            self._backend = None
