"""Unit tests for DuplicateRowStatistic."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd


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

DataType = importlib.import_module("iaml.data_type").DataType
from tests.statistics.statistic_test_case import StatisticTestCase

DuplicateRowStatistic = _load_module(
    "iaml.statistics.duplicate_row_statistic",
    IAML_PATH / "statistics" / r"duplicate_row_statistic.py",
).DuplicateRowStatistic


class TestDuplicateRowStatistic(StatisticTestCase):
    """Coverage for DuplicateRowStatistic core behavior."""

    def test_classification_duplicates_ignore_text(self) -> None:
        dataset = self.make_classification_dataset(n_samples=18, seed=6, n_classes=3)
        non_text_columns = [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]
        text_columns = [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] in (DataType.TEXT, DataType.SHORT_TEXT)
        ]
        self.assertTrue(non_text_columns)
        self.assertTrue(text_columns)

        dataset.X.loc[1, non_text_columns] = dataset.X.loc[0, non_text_columns]
        for col in text_columns:
            dataset.X.loc[1, col] = f"changed_{col}"

        full_duplicates = int(dataset.X.duplicated().sum())
        self.assertEqual(full_duplicates, 0)

        result = self.compute_statistic(DuplicateRowStatistic, dataset)

        expected_duplicates = int(dataset.X[non_text_columns].duplicated().sum())
        expected_ratio = expected_duplicates / len(dataset.X)
        expected = pd.DataFrame(
            [[expected_duplicates, expected_ratio]],
            index=["duplicate_rows"],
            columns=["n_duplicate_rows", "ratio_duplicate_rows"],
        )
        self.assertFrameEqual(result, expected)

    def test_text_only_columns_report_zero_duplicates(self) -> None:
        base = self.make_classification_dataset(n_samples=10, seed=3, n_classes=2)
        X_text = base.X[["short_text", "long_text"]].copy()
        X_text.loc[1] = X_text.loc[0]
        dataset = self.make_dataset(X_text, base.y)

        result = self.compute_statistic(DuplicateRowStatistic, dataset)

        expected = pd.DataFrame(
            [[0, 0.0]],
            index=["duplicate_rows"],
            columns=["n_duplicate_rows", "ratio_duplicate_rows"],
        )
        self.assertFrameEqual(result, expected)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=15)
        result = self.compute_statistic(DuplicateRowStatistic, dataset)

        self.assertTrue(result.empty)
