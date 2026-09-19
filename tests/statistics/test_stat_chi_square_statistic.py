"""Unit tests for ChiSquareStatistic."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency


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


def _chi2_statistic(feature: pd.Series, target: pd.Series) -> float | None:
    mask = feature.notna() & target.notna()
    if not mask.any():
        return None
    table = pd.crosstab(feature[mask], target[mask])
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    stat, _, _, _ = chi2_contingency(table, correction=False)
    if pd.isna(stat):
        return None
    return float(stat)


_ensure_package("iaml", IAML_PATH)
_ensure_package("iaml.statistics", IAML_PATH / "statistics")

from tests.statistics.statistic_test_case import StatisticTestCase

DataType = _load_module("iaml.data_type", IAML_PATH / "data_type.py").DataType

ChiSquareStatistic = _load_module(
    "iaml.statistics.chi_square_statistic",
    IAML_PATH / "statistics" / r"chi_square_statistic.py",
).ChiSquareStatistic


class TestChiSquareStatistic(StatisticTestCase):
    """Coverage for ChiSquareStatistic core behavior."""

    @staticmethod
    def _force_series_target(dataset) -> None:
        y = dataset.y
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError("Expected 1D target for series conversion.")
            y = y.iloc[:, 0]
        dataset._Dataset__y = pd.Series(y)

    def test_classification_columns_and_values(self) -> None:
        dataset = self.make_classification_dataset(n_samples=45, seed=12, n_classes=3)
        self._force_series_target(dataset)
        result = self.compute_statistic(ChiSquareStatistic, dataset)

        self.assertEqual(set(result.index), {"chi_square"})

        categorical_columns = dataset.get_columns_names_by_type(DataType.CATEGORICAL)
        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in categorical_columns
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        expected_all = _chi2_statistic(dataset.X["cat_small"], dataset.y)
        value_all = result.loc["chi_square", "cat_small_all"]
        if expected_all is None:
            self.assertTrue(pd.isna(value_all))
        else:
            self.assertAlmostEqual(value_all, expected_all)

        sample_label = class_labels[0]
        expected_label = _chi2_statistic(
            dataset.X["cat_small"], dataset.y == sample_label
        )
        value_label = result.loc["chi_square", f"cat_small_{sample_label}"]
        if expected_label is None:
            self.assertTrue(pd.isna(value_label))
        else:
            self.assertAlmostEqual(value_label, expected_label)

        self.assertNotIn("num_1_all", result.columns)
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
                result = self.compute_statistic(ChiSquareStatistic, dataset)
                self.assertTrue(result.empty)
