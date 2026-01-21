"""Unit tests for CorrelationWithTargetStatistic."""
from __future__ import annotations

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

DataType = _load_module("iaml.data_type", IAML_PATH / "data_type.py").DataType
_load_module("iaml.statistic", IAML_PATH / "statistic.py")

CorrelationWithTargetStatistic = _load_module(
    "iaml.statistics.correlation_with_target",
    IAML_PATH / "statistics" / r"correlation_with_target.py",
).CorrelationWithTargetStatistic

from tests.statistics.statistic_test_case import StatisticTestCase


def _eta(feature: pd.Series, target: pd.Series) -> float | None:
    mask = feature.notna() & target.notna()
    if not mask.any():
        return None
    values = feature[mask]
    groups = target[mask]
    if groups.nunique() < 2:
        return None
    overall_mean = values.mean()
    ss_total = ((values - overall_mean) ** 2).sum()
    if ss_total == 0:
        return None
    ss_between = 0.0
    for _, group_values in values.groupby(groups):
        count = group_values.size
        if count == 0:
            continue
        mean = group_values.mean()
        ss_between += count * (mean - overall_mean) ** 2
    eta_sq = ss_between / ss_total
    if pd.isna(eta_sq):
        return None
    return float(eta_sq) ** 0.5


class TestCorrelationWithTargetStatistic(StatisticTestCase):
    """Coverage for CorrelationWithTargetStatistic core behavior."""

    def _set_series_target(self, dataset) -> None:
        dataset._Dataset__y = pd.Series(dataset.y)  # pylint: disable=protected-access

    def test_continuous_correlations_for_numeric_columns(self) -> None:
        dataset = self.make_regression_dataset(n_samples=25, seed=31)
        self._set_series_target(dataset)
        result = self.compute_statistic(CorrelationWithTargetStatistic, dataset)

        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        self.assertEqual(set(result.index), {"correlation_with_target"})
        self.assertEqual(set(result.columns), set(numeric_columns))
        self.assertNotIn("cat_small", result.columns)

        expected = dataset.X["num_1"].corr(pd.Series(dataset.y))
        self.assertAlmostEqual(result.loc["correlation_with_target", "num_1"], expected)

    def test_classification_eta_and_columns(self) -> None:
        dataset = self.make_classification_dataset(n_samples=30, seed=22, n_classes=3)
        self._set_series_target(dataset)
        result = self.compute_statistic(CorrelationWithTargetStatistic, dataset)

        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in numeric_columns
            for label in ["all"] + class_labels
        }

        self.assertEqual(set(result.columns), expected_columns)
        self.assertNotIn("cat_small_all", result.columns)

        overall_expected = _eta(dataset.X["num_1"], pd.Series(dataset.y))
        overall_value = result.loc["correlation_with_target", "num_1_all"]
        if overall_expected is None:
            self.assertTrue(pd.isna(overall_value))
        else:
            self.assertAlmostEqual(overall_value, overall_expected)

        sample_label = class_labels[0]
        label_expected = _eta(dataset.X["num_1"], pd.Series(dataset.y == sample_label))
        label_value = result.loc[
            "correlation_with_target",
            f"num_1_{sample_label}",
        ]
        if label_expected is None:
            self.assertTrue(pd.isna(label_value))
        else:
            self.assertAlmostEqual(label_value, label_expected)

    def test_survival_returns_empty(self) -> None:
        dataset = self.make_survival_dataset(n_samples=12, seed=41)
        result = self.compute_statistic(CorrelationWithTargetStatistic, dataset)

        self.assertTrue(result.empty)
