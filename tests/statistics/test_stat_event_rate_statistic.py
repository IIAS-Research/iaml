"""Unit tests for EventRateStatistic."""
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

EventRateStatistic = _load_module(
    "iaml.statistics.event_rate_statistic",
    IAML_PATH / "statistics" / "event_rate_statistic.py",
).EventRateStatistic

from tests.statistics.statistic_test_case import StatisticTestCase


class TestEventRateStatistic(StatisticTestCase):
    """Coverage for EventRateStatistic core behavior."""

    def test_survival_event_and_censor_rates(self) -> None:
        dataset = self.make_survival_dataset(n_samples=25, seed=5)
        result = self.compute_statistic(EventRateStatistic, dataset)

        columns = ["event_count", "censor_count", "event_rate", "censor_rate"]
        self.assertEqual(list(result.index), ["event_rate"])
        self.assertEqual(list(result.columns), columns)

        event_count = sum(1 for event, _ in dataset.y if event)
        n_samples = len(dataset.y)
        censor_count = n_samples - event_count
        expected = pd.DataFrame(
            [[event_count, censor_count, event_count / n_samples, censor_count / n_samples]],
            index=["event_rate"],
            columns=columns,
        )

        self.assertFrameEqual(result, expected)

    def test_survival_empty_dataset_returns_zero_rates(self) -> None:
        dataset = self.make_survival_dataset(n_samples=0, seed=11)
        result = self.compute_statistic(EventRateStatistic, dataset)

        expected = pd.DataFrame(
            [[0, 0, 0.0, 0.0]],
            index=["event_rate"],
            columns=["event_count", "censor_count", "event_rate", "censor_rate"],
        )
        self.assertFrameEqual(result, expected)

    def test_non_survival_returns_empty(self) -> None:
        dataset = self.make_regression_dataset(n_samples=20, seed=9)
        result = self.compute_statistic(EventRateStatistic, dataset)

        self.assertTrue(result.empty)
