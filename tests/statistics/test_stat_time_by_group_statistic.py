"""Unit tests for TimeByGroupStatistic."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IAML_PATH = PROJECT_ROOT / "src" / "iaml"
STATISTICS_PATH = IAML_PATH / "statistics"


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
_ensure_package("iaml.statistics", STATISTICS_PATH)

TimeByGroupStatistic = _load_module(
    "iaml.statistics.time_by_group_statistic",
    STATISTICS_PATH / r"time_by_group_statistic.py",
).TimeByGroupStatistic

from iaml.dataset import Dataset
from tests.helpers.datasets import make_statistics_survival_data
from tests.statistics.statistic_test_case import StatisticTestCase


class TestTimeByGroupStatistic(StatisticTestCase):
    """Coverage for TimeByGroupStatistic core behavior."""

    def test_survival_group_event_rate_and_median_time(self) -> None:
        X, y = make_statistics_survival_data(n_samples=40, seed=11)
        X["cat_dtype"] = np.where(X["num_1"] > 0, "pos", "neg")
        dataset = Dataset(X, y)
        statistic = TimeByGroupStatistic()

        result = statistic.compute(dataset)

        self.assertEqual(list(result.index), ["time_by_group"])
        self.assertFalse(result.empty)

        columns = statistic._select_columns(dataset)
        self.assertIn("cat_small", columns)
        self.assertIn("cat_with_nan", columns)
        self.assertIn("cat_dtype", columns)

        samples = Dataset.normalize_survival_target(dataset.y)
        events = np.asarray([event for event, _ in samples], dtype=float)
        times = np.asarray([time for _, time in samples], dtype=float)

        frame = dataset.X[columns].copy()
        frame["_event"] = events
        frame["_time"] = times

        for col in ["cat_small", "cat_dtype"]:
            values = frame[[col, "_event", "_time"]].dropna(subset=[col])
            grouped = values.groupby(col, sort=True)
            event_rates = grouped["_event"].mean()
            median_times = grouped["_time"].median()
            for group_value in event_rates.index:
                value_label = str(group_value)
                event_col = f"{col}_{value_label}_event_rate"
                median_col = f"{col}_{value_label}_median_time"
                self.assertIn(event_col, result.columns)
                self.assertIn(median_col, result.columns)
                self.assertAlmostEqual(
                    result.loc["time_by_group", event_col],
                    float(event_rates.loc[group_value]),
                )
                self.assertAlmostEqual(
                    result.loc["time_by_group", median_col],
                    float(median_times.loc[group_value]),
                )

    def test_non_survival_returns_empty(self) -> None:
        dataset = self.make_regression_dataset(n_samples=20, seed=9)
        result = self.compute_statistic(TimeByGroupStatistic, dataset)

        self.assertTrue(result.empty)

    def test_survival_without_categorical_returns_empty(self) -> None:
        X, y = make_statistics_survival_data(n_samples=18, seed=12)
        dataset = Dataset(X[["num_1", "num_2", "num_3"]], y)
        result = self.compute_statistic(TimeByGroupStatistic, dataset)

        self.assertTrue(result.empty)
