"""Unit tests for DataTypeSummaryStatistic."""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from enum import Enum
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAT_PATH = PROJECT_ROOT / "src" / "iaml" / "statistics" / r"data_type_summary_statistic.py"


def _install_stub_modules() -> None:
    if "iaml" not in sys.modules:
        iaml_pkg = types.ModuleType("iaml")
        iaml_pkg.__path__ = [str(PROJECT_ROOT / "src" / "iaml")]
        sys.modules["iaml"] = iaml_pkg

    if "iaml.statistics" not in sys.modules:
        stats_pkg = types.ModuleType("iaml.statistics")
        stats_pkg.__path__ = [str(PROJECT_ROOT / "src" / "iaml" / "statistics")]
        sys.modules["iaml.statistics"] = stats_pkg

    if "iaml.data_type" not in sys.modules:
        data_type_mod = types.ModuleType("iaml.data_type")

        class DataType(Enum):
            CATEGORICAL = 0
            TEXT = 1
            SHORT_TEXT = 2
            NUMERIC = 3
            DATE = 4

        data_type_mod.DataType = DataType
        sys.modules["iaml.data_type"] = data_type_mod

    if "iaml.dataset" not in sys.modules:
        dataset_mod = types.ModuleType("iaml.dataset")

        class Dataset:
            def __init__(self, X: pd.DataFrame, type_of_target: str, columns_by_type: dict):
                self.X = X
                self.type_of_target = type_of_target
                self._columns_by_type = columns_by_type

            def get_columns_names_by_type(self, data_types):
                if not isinstance(data_types, (list, tuple, set)):
                    data_types = [data_types]
                names = []
                for data_type in data_types:
                    names.extend(self._columns_by_type.get(data_type, []))
                return names

        dataset_mod.Dataset = Dataset
        sys.modules["iaml.dataset"] = dataset_mod

    if "iaml.statistic" not in sys.modules:
        statistic_mod = types.ModuleType("iaml.statistic")

        class Statistic:
            def compute(self, dataset, **kwargs):
                raise NotImplementedError

            def suitable(self, dataset):
                return False

        statistic_mod.Statistic = Statistic
        sys.modules["iaml.statistic"] = statistic_mod


def _load_statistic():
    _install_stub_modules()
    module_name = "iaml.statistics.data_type_summary_statistic"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, STAT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {STAT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


DataTypeSummaryStatistic = _load_statistic().DataTypeSummaryStatistic
DataType = sys.modules["iaml.data_type"].DataType
Dataset = sys.modules["iaml.dataset"].Dataset


class TestDataTypeSummaryStatistic(unittest.TestCase):
    """Coverage for DataTypeSummaryStatistic core behavior."""

    def assertFrameEqual(self, left: pd.DataFrame, right: pd.DataFrame) -> None:
        try:
            assert_frame_equal(left, right)
        except AssertionError as exc:
            self.fail(str(exc))

    def _expected_summary(self, dataset) -> pd.DataFrame:
        total_columns = len(dataset.X.columns)
        n_numeric = len(dataset.get_columns_names_by_type(DataType.NUMERIC))
        n_categorical = len(dataset.get_columns_names_by_type(DataType.CATEGORICAL))
        n_text = len(dataset.get_columns_names_by_type([DataType.TEXT, DataType.SHORT_TEXT]))
        n_date = len(dataset.get_columns_names_by_type(DataType.DATE))

        if total_columns:
            ratio_numeric = n_numeric / total_columns
            ratio_categorical = n_categorical / total_columns
            ratio_text = n_text / total_columns
            ratio_date = n_date / total_columns
        else:
            ratio_numeric = 0.0
            ratio_categorical = 0.0
            ratio_text = 0.0
            ratio_date = 0.0

        data = [[
            n_numeric,
            n_categorical,
            n_text,
            n_date,
            ratio_numeric,
            ratio_categorical,
            ratio_text,
            ratio_date,
        ]]
        columns = [
            "n_numeric",
            "n_categorical",
            "n_text",
            "n_date",
            "ratio_numeric",
            "ratio_categorical",
            "ratio_text",
            "ratio_date",
        ]
        return pd.DataFrame(data, index=["data_type_summary"], columns=columns)

    def test_counts_and_ratios(self) -> None:
        columns = ["num_1", "cat_1", "text_1", "short_text_1", "date_1"]
        X = pd.DataFrame(columns=columns)
        columns_by_type = {
            DataType.NUMERIC: ["num_1"],
            DataType.CATEGORICAL: ["cat_1"],
            DataType.TEXT: ["text_1"],
            DataType.SHORT_TEXT: ["short_text_1"],
            DataType.DATE: ["date_1"],
        }
        dataset = Dataset(X, "classification", columns_by_type)
        statistic = DataTypeSummaryStatistic()
        result = statistic.compute(dataset)

        expected = self._expected_summary(dataset)
        self.assertFrameEqual(result, expected)

    def test_survival_returns_empty(self) -> None:
        X = pd.DataFrame(columns=["num_1"])
        dataset = Dataset(X, "survival", {DataType.NUMERIC: ["num_1"]})
        statistic = DataTypeSummaryStatistic()
        result = statistic.compute(dataset)

        self.assertTrue(result.empty)
