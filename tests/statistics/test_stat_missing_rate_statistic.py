"""Unit tests for MissingRateStatistic."""
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

MissingRateStatistic = _load_module(
    "iaml.statistics.missing_rate_statistic",
    IAML_PATH / "statistics" / r"missing_rate_statistic.py",
).MissingRateStatistic

from tests.statistics.statistic_test_case import StatisticTestCase


class TestMissingRateStatistic(StatisticTestCase):
    """Coverage for MissingRateStatistic core behavior."""

    def _non_text_columns(self, dataset) -> list[str]:
        return [
            col
            for col in dataset.X.columns
            if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
        ]

    def test_classification_missing_rates_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=36, seed=14, n_classes=3)
        result = self.compute_statistic(MissingRateStatistic, dataset)

        self.assertEqual(list(result.index), ["missing_rate"])

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in self._non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        expected_all = dataset.X["num_2"].isna().mean()
        self.assertAlmostEqual(result.loc["missing_rate", "num_2_all"], expected_all)

        sample_label = class_labels[0]
        expected_label = dataset.X.loc[dataset.y == sample_label, "num_2"].isna().mean()
        self.assertAlmostEqual(
            result.loc["missing_rate", f"num_2_{sample_label}"], expected_label
        )

        for col in dataset.X.columns:
            if dataset.columns_types[col][1] in (DataType.TEXT, DataType.SHORT_TEXT):
                self.assertFalse(any(name.startswith(f"{col}_") for name in result.columns))

    def test_continuous_missing_rates_exclude_text(self) -> None:
        dataset = self.make_regression_dataset(n_samples=28, seed=19)
        result = self.compute_statistic(MissingRateStatistic, dataset)

        self.assertEqual(list(result.index), ["missing_rate"])
        self.assertEqual(set(result.columns), set(self._non_text_columns(dataset)))

        expected_all = dataset.X["num_2"].isna().mean()
        self.assertAlmostEqual(result.loc["missing_rate", "num_2"], expected_all)
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=20, seed=27)
        result = self.compute_statistic(MissingRateStatistic, dataset)

        self.assertTrue(result.empty)
