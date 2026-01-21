"""Unit tests for TimeSummaryStatistic."""
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


def _find_statistic_module(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No statistic module matching {pattern} in {path}")
    return matches[0]


_ensure_package("iaml", IAML_PATH)
_ensure_package("iaml.statistics", STATISTICS_PATH)

TimeSummaryStatistic = _load_module(
    "iaml.statistics.time_summary_statistic",
    _find_statistic_module(STATISTICS_PATH, "tim*tatistic.py"),
).TimeSummaryStatistic

from iaml.dataset import Dataset
from tests.statistics.statistic_test_case import StatisticTestCase


class TestTimeSummaryStatistic(StatisticTestCase):
    """Coverage for TimeSummaryStatistic core behavior."""

    def test_survival_summary_values(self) -> None:
        dataset = self.make_survival_dataset(n_samples=25, seed=7)
        result = self.compute_statistic(TimeSummaryStatistic, dataset)

        samples = Dataset.normalize_survival_target(dataset.y)
        times = np.asarray([time for _, time in samples], dtype=float)

        columns = ["time_min", "time_median", "time_max"]
        columns += [f"time_quantile_{q}" for q in TimeSummaryStatistic.quantiles]
        expected = pd.DataFrame(
            [
                [
                    float(np.min(times)),
                    float(np.median(times)),
                    float(np.max(times)),
                ]
                + np.quantile(times, TimeSummaryStatistic.quantiles).astype(float).tolist()
            ],
            index=["time_summary"],
            columns=columns,
        )

        self.assertFrameEqual(result, expected)

    def test_survival_empty_times(self) -> None:
        dataset = Dataset(pd.DataFrame(), [])
        result = self.compute_statistic(TimeSummaryStatistic, dataset)

        columns = ["time_min", "time_median", "time_max"]
        columns += [f"time_quantile_{q}" for q in TimeSummaryStatistic.quantiles]
        expected = pd.DataFrame(
            [[None, None, None] + [None] * len(TimeSummaryStatistic.quantiles)],
            index=["time_summary"],
            columns=columns,
        )

        self.assertFrameEqual(result, expected)

    def test_non_survival_returns_empty(self) -> None:
        dataset = self.make_regression_dataset(n_samples=20, seed=3)
        result = self.compute_statistic(TimeSummaryStatistic, dataset)

        self.assertTrue(result.empty)
