"""Unit tests for EffectSizeStatistic."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np
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


def _effect_size_module_path() -> Path:
    matches = sorted((IAML_PATH / "statistics").glob("effec*tatistic.py"))
    if len(matches) != 1:
        raise FileNotFoundError("Expected a single effect size statistic module")
    return matches[0]


EffectSizeStatistic = _load_module(
    "iaml.statistics.effect_size_statistic",
    _effect_size_module_path(),
).EffectSizeStatistic


def _make_classification_dataset(n_samples: int, seed: int, n_classes: int):
    X, y = make_statistics_classification_data(
        n_samples=n_samples, seed=seed, n_classes=n_classes
    )
    return Dataset(X, y)


def _make_regression_dataset(n_samples: int, seed: int):
    X, y = make_statistics_regression_data(n_samples=n_samples, seed=seed)
    return Dataset(X, y)


def _make_survival_dataset(n_samples: int, seed: int):
    X, y = make_statistics_survival_data(n_samples=n_samples, seed=seed)
    return Dataset(X, y)


def _safe_variance(values: pd.Series) -> float | None:
    if len(values) <= 1:
        return None
    var = values.var(ddof=1)
    if pd.isna(var) or var < 0:
        return None
    return float(var)


def _cohen_d(values_a: pd.Series, values_b: pd.Series) -> float | None:
    values_a = values_a.dropna()
    values_b = values_b.dropna()
    n_a = float(len(values_a))
    n_b = float(len(values_b))
    if n_a <= 1 or n_b <= 1:
        return None
    mean_a = values_a.mean()
    mean_b = values_b.mean()
    if pd.isna(mean_a) or pd.isna(mean_b):
        return None
    var_a = _safe_variance(values_a)
    var_b = _safe_variance(values_b)
    if var_a is None or var_b is None:
        return None
    pooled_denom = n_a + n_b - 2.0
    if pooled_denom <= 0:
        return None
    pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / pooled_denom
    if pooled_var <= 0 or pd.isna(pooled_var):
        return None
    pooled_std = np.sqrt(pooled_var)
    if pooled_std == 0 or pd.isna(pooled_std):
        return None
    return (mean_a - mean_b) / pooled_std


class TestEffectSizeStatistic(unittest.TestCase):
    """Coverage for EffectSizeStatistic core behavior."""

    def test_classification_effect_sizes_and_columns(self) -> None:
        dataset = _make_classification_dataset(n_samples=36, seed=11, n_classes=3)
        result = EffectSizeStatistic().compute(dataset)

        self.assertEqual(set(result.index), {"effect_size"})

        class_labels = list(pd.unique(dataset.y))
        numeric_columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        expected_columns = {
            f"{col}_{label}"
            for col in numeric_columns
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        sample_label = class_labels[0]
        values_a = dataset.X.loc[dataset.y == sample_label, "num_1"]
        values_b = dataset.X.loc[dataset.y != sample_label, "num_1"]
        expected_label = _cohen_d(values_a, values_b)
        label_value = result.loc["effect_size", f"num_1_{sample_label}"]
        if expected_label is None:
            self.assertTrue(pd.isna(label_value))
        else:
            self.assertAlmostEqual(label_value, expected_label)

        abs_effects = []
        for label in class_labels:
            values_a = dataset.X.loc[dataset.y == label, "num_1"]
            values_b = dataset.X.loc[dataset.y != label, "num_1"]
            d_value = _cohen_d(values_a, values_b)
            if d_value is not None and not pd.isna(d_value):
                abs_effects.append(abs(d_value))
        if abs_effects:
            expected_all = float(np.mean(abs_effects))
            self.assertAlmostEqual(result.loc["effect_size", "num_1_all"], expected_all)

        for col in ["cat_small", "date_col", "short_text", "long_text"]:
            self.assertFalse(any(name.startswith(f"{col}_") for name in result.columns))

    def test_continuous_returns_empty(self) -> None:
        dataset = _make_regression_dataset(n_samples=20, seed=31)
        result = EffectSizeStatistic().compute(dataset)

        self.assertTrue(result.empty)

    def test_survival_returns_empty(self) -> None:
        dataset = _make_survival_dataset(n_samples=20, seed=41)
        result = EffectSizeStatistic().compute(dataset)

        self.assertTrue(result.empty)
