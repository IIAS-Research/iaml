"""Unit tests for CoefVariationStatistic."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
import unittest

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

CoefVariationStatistic = _load_module(
    "iaml.statistics.coef_variation_statistic",
    IAML_PATH / "statistics" / r"coef_variation_statistic.py",
).CoefVariationStatistic


def _non_text_columns(dataset: Dataset) -> list[str]:
    return [
        col
        for col in dataset.X.columns
        if dataset.columns_types[col][1] not in (DataType.TEXT, DataType.SHORT_TEXT)
    ]


def _coef_variation(values: pd.Series) -> float | None:
    mean = values.mean()
    if pd.isna(mean) or mean == 0:
        return None
    stdev = values.std()
    if pd.isna(stdev):
        return None
    return stdev / mean


class TestCoefVariationStatistic(unittest.TestCase):
    """Coverage for CoefVariationStatistic core behavior."""

    def test_classification_coefficients_and_columns(self) -> None:
        X, y = make_statistics_classification_data(n_samples=18, seed=29, n_classes=3)
        dataset = Dataset(X, y)
        result = CoefVariationStatistic().compute(dataset)

        self.assertEqual(set(result.index), {"coef_variation"})

        class_labels = list(pd.unique(dataset.y))
        expected_columns = {
            f"{col}_{label}"
            for col in _non_text_columns(dataset)
            for label in ["all"] + class_labels
        }
        self.assertEqual(set(result.columns), expected_columns)

        all_expected = _coef_variation(dataset.X["num_2"])
        all_value = result.loc["coef_variation", "num_2_all"]
        if all_expected is None:
            self.assertTrue(pd.isna(all_value))
        else:
            self.assertAlmostEqual(all_value, all_expected)

        sample_label = class_labels[0]
        label_expected = _coef_variation(
            dataset.X.loc[dataset.y == sample_label, "num_2"]
        )
        label_value = result.loc["coef_variation", f"num_2_{sample_label}"]
        if label_expected is None:
            self.assertTrue(pd.isna(label_value))
        else:
            self.assertAlmostEqual(label_value, label_expected)

        for label in ["all"] + class_labels:
            self.assertTrue(pd.isna(result.loc["coef_variation", f"cat_small_{label}"]))
            self.assertTrue(pd.isna(result.loc["coef_variation", f"date_col_{label}"]))
            self.assertNotIn(f"short_text_{label}", result.columns)
            self.assertNotIn(f"long_text_{label}", result.columns)

    def test_continuous_coefficients_exclude_text_columns(self) -> None:
        X, y = make_statistics_regression_data(n_samples=16, seed=39)
        dataset = Dataset(X, y)
        result = CoefVariationStatistic().compute(dataset)

        self.assertEqual(set(result.index), {"coef_variation"})
        self.assertEqual(set(result.columns), set(_non_text_columns(dataset)))

        expected = _coef_variation(dataset.X["num_2"])
        value = result.loc["coef_variation", "num_2"]
        if expected is None:
            self.assertTrue(pd.isna(value))
        else:
            self.assertAlmostEqual(value, expected)
        self.assertTrue(pd.isna(result.loc["coef_variation", "cat_small"]))
        self.assertTrue(pd.isna(result.loc["coef_variation", "date_col"]))
        self.assertNotIn("short_text", result.columns)
        self.assertNotIn("long_text", result.columns)

    def test_survival_returns_empty(self) -> None:
        X, y = make_statistics_survival_data(n_samples=10, seed=47)
        dataset = Dataset(X, y)
        result = CoefVariationStatistic().compute(dataset)

        self.assertTrue(result.empty)
