"""Unit tests for SummaryTableStatistic."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

from tests.helpers.datasets import (
    make_statistics_classification_data,
    make_statistics_regression_data,
    make_statistics_survival_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IAML_PATH = PROJECT_ROOT / "src" / "iaml"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("iaml", IAML_PATH)
_ensure_package("iaml.statistics", IAML_PATH / "statistics")

DataType = _load_module("iaml.data_type", IAML_PATH / "data_type.py").DataType
Dataset = _load_module("iaml.dataset", IAML_PATH / "dataset.py").Dataset
_load_module("iaml.statistic", IAML_PATH / "statistic.py")

SummaryTableStatistic = _load_module(
    "iaml.statistics.summary_table_statistic",
    IAML_PATH / "statistics" / r"summary_table_statistic.py",
).SummaryTableStatistic


class TestSummaryTableStatistic(unittest.TestCase):
    """Coverage for SummaryTableStatistic core behavior."""

    def _make_dataset(self, factory, n_samples: int, seed: int, n_classes: int | None = None):
        if n_classes is None:
            X, y = factory(n_samples=n_samples, seed=seed)
        else:
            X, y = factory(n_samples=n_samples, seed=seed, n_classes=n_classes)
        return Dataset(X, y)

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def _expected_summary(self, dataset) -> pd.DataFrame:
        columns = self._non_text_columns(dataset)
        if columns:
            data_frame = dataset.X[columns]
            n_missing_total = int(data_frame.isna().sum().sum())
            memory = int(data_frame.memory_usage(deep=True).sum())
        else:
            n_missing_total = 0
            memory = 0

        return pd.DataFrame(
            [[dataset.X.shape[0], len(columns), n_missing_total, memory]],
            index=["summary_table"],
            columns=["n_rows", "n_cols", "n_missing_total", "memory"],
        )

    def test_classification_summary_matches_expected(self) -> None:
        dataset = self._make_dataset(
            make_statistics_classification_data, n_samples=25, seed=5, n_classes=3
        )
        result = SummaryTableStatistic().compute(dataset)

        expected = self._expected_summary(dataset)
        pd.testing.assert_frame_equal(result, expected)

    def test_regression_summary_matches_expected(self) -> None:
        dataset = self._make_dataset(make_statistics_regression_data, n_samples=20, seed=9)
        result = SummaryTableStatistic().compute(dataset)

        expected = self._expected_summary(dataset)
        pd.testing.assert_frame_equal(result, expected)

    def test_survival_returns_empty(self) -> None:
        dataset = self._make_dataset(make_statistics_survival_data, n_samples=12, seed=21)
        result = SummaryTableStatistic().compute(dataset)

        self.assertTrue(result.empty)
