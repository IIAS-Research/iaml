"""[STATISTIC] Parent of descriptive statistics."""
from __future__ import annotations

from typing import Any
import pandas as pd

from .dataset import Dataset
from .reference import Reference


class Statistic:
    """[STATISTIC] Parent of descriptive statistics."""

    name: str = ""
    """Name of the statistic."""

    _description: str = ""
    """Short description of the statistic."""

    _description_long: str = ""
    """Long description of the statistic."""

    refs: list[dict[str, Any]] = []
    """List of references for this statistic."""

    @classmethod
    def all_subclasses(cls) -> list['Statistic']:
        """Return all statistic subclasses."""
        subclasses = cls.__subclasses__()
        for subclass in subclasses:
            subclasses += subclass.all_subclasses()
        return subclasses

    @classmethod
    def get_refs(cls) -> list[Reference]:
        """Get bibliography references."""
        if hasattr(cls, 'refs'):
            return [Reference(ref, cls.__name__) for ref in cls.refs]
        return []

    @property
    def description(self) -> str:
        """Return the short description."""
        return self._description.replace('\n', '')

    @property
    def description_long(self) -> str:
        """Return the long description."""
        return self._description_long.replace('\n', '')

    def explain(self) -> str:
        """Describe statistic."""
        return self.description_long

    def compute(self, dataset: Dataset, **kwargs) -> pd.DataFrame:
        """Compute the statistic on the dataset."""
        raise NotImplementedError('Subclass must implement abstract method')

    def suitable(self, dataset: Dataset) -> bool:  # pylint: disable=unused-argument
        """Return True if the statistic applies to the dataset."""
        return False

    @property
    def name(self) -> str:
        """Return the statistic formatted name."""
        return ' '.join(x.title() for x in str(self).split('_'))
