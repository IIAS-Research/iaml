"""Unit tests for ANOVAStatistic."""
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

ANOVAStatistic = _load_module(
    "iaml.statistics.anova_statistic",
    IAML_PATH / "statistics" / "anova_statistic.py",
).ANOVAStatistic


class TestANOVAStatistic(StatisticTestCase):
    """Coverage for ANOVAStatistic core behavior."""

    def _numeric_columns(self, dataset) -> list[str]:
        return dataset.get_columns_names_by_type(DataType.NUMERIC)

    def test_classification_anova_columns_and_values(self) -> None:
        dataset = self.make_classification_dataset(n_samples=60, seed=21, n_classes=3)
        result = self.compute_statistic(ANOVAStatistic, dataset)

        self.assertEqual(set(result.index), {"anova"})

        class_labels = list(pd.unique(dataset.y))
        numeric_columns = self._numeric_columns(dataset)
        expected_columns = {
            f"{col}_{label}"
            for col in numeric_columns
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        for label in ["all"] + class_labels:
            value = result.loc["anova", f"num_1_{label}"]
            self.assertFalse(pd.isna(value))
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

        self.assertNotIn("cat_small_all", result.columns)
        self.assertNotIn("date_col_all", result.columns)
        self.assertNotIn("short_text_all", result.columns)
        self.assertNotIn("long_text_all", result.columns)

    def test_non_classification_returns_empty(self) -> None:
        datasets = [
            ("continuous", self.make_regression_dataset(n_samples=20, seed=31)),
            ("survival", self.make_survival_dataset(n_samples=20, seed=41)),
        ]

        for name, dataset in datasets:
            with self.subTest(name=name):
                result = self.compute_statistic(ANOVAStatistic, dataset)
                self.assertTrue(result.empty)
