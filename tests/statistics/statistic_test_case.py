"""Base helpers for statistic tests."""
from __future__ import annotations

from typing import Iterable, Type
import sys
from pathlib import Path
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

# Ensure src/ is importable when running tests from the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from iaml.dataset import Dataset
from iaml.statistic import Statistic

from tests.helpers.datasets import (
    make_statistics_classification_data,
    make_statistics_regression_data,
    make_statistics_survival_data,
)


class StatisticTestCase(unittest.TestCase):
    """Shared helpers to make writing statistic tests concise."""

    def make_dataset(self, X: pd.DataFrame, y: Iterable | None = None) -> Dataset:
        if y is None:
            y = [0] * len(X)
        return Dataset(X, y)

    def make_classification_dataset(
        self,
        n_samples: int = 120,
        seed: int = 20,
        n_classes: int = 2,
    ) -> Dataset:
        X, y = make_statistics_classification_data(
            n_samples=n_samples, seed=seed, n_classes=n_classes
        )
        return Dataset(X, y)

    def make_regression_dataset(
        self,
        n_samples: int = 120,
        seed: int = 30,
    ) -> Dataset:
        X, y = make_statistics_regression_data(n_samples=n_samples, seed=seed)
        return Dataset(X, y)

    def make_survival_dataset(
        self,
        n_samples: int = 120,
        seed: int = 40,
    ) -> Dataset:
        X, y = make_statistics_survival_data(n_samples=n_samples, seed=seed)
        return Dataset(X, y)

    def compute_statistic(
        self,
        statistic_cls: Type[Statistic],
        dataset: Dataset,
    ) -> pd.DataFrame:
        statistic = statistic_cls()
        return statistic.compute(dataset)

    def compute_all_statistics(self, dataset: Dataset) -> pd.DataFrame:
        """Compute all suitable statistics for a dataset."""
        computed_statistics = pd.DataFrame()
        for statistic_cls in Statistic.all_subclasses():
            statistic = statistic_cls()
            if statistic.suitable(dataset):
                result = statistic.compute(dataset)
                if result is not None and not result.empty:
                    computed_statistics = pd.concat([computed_statistics, result])
        return computed_statistics

    def assertFrameEqual(self, left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> None:
        try:
            assert_frame_equal(left, right, **kwargs)
        except AssertionError as exc:
            self.fail(str(exc))
