"""Contracts for descriptive statistic and plot tests."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type
import sys
from pathlib import Path

import pandas as pd

# Ensure src/ is importable when running tests from the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from iaml.dataset import Dataset
from iaml.plot import StatisticPlot
from iaml.statistic import Statistic
from iaml.plots import BarPlot, LinePlot
from iaml.statistics import (
    BoundStatistic,
    CountStatistic,
    MeanStatistic,
    ModeStatistic,
    QuantileStatistic,
    RangeStatistic,
    StdevStatistic,
    ValueCountsStatistic,
    VarianceStatistic,
    ViolinStatistic,
)
from iaml.statistics.kurtosis import KurtosisStatistic
from iaml.statistics.skewness import SkewnessStatistic

from tests.helpers.datasets import (
    make_statistics_classification_data,
    make_statistics_regression_data,
    make_statistics_survival_data,
)


def _dataset_from_factory(factory: Callable[[], tuple]) -> Dataset:
    X, y = factory()
    return Dataset(X, y)


DATASET_FACTORIES: dict[str, Callable[[], tuple]] = {
    "classification": make_statistics_classification_data,
    "continuous": make_statistics_regression_data,
    "survival": make_statistics_survival_data,
}


@dataclass(frozen=True)
class StatisticContract:
    statistic_cls: Type[Statistic]
    targets: tuple[str, ...]
    expected_rows: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlotContract:
    plot_cls: Type[StatisticPlot]
    targets: tuple[str, ...]
    required_rows: tuple[str, ...]
    notes: tuple[str, ...] = ()


STATISTIC_CONTRACTS: tuple[StatisticContract, ...] = (
    StatisticContract(
        statistic_cls=CountStatistic,
        targets=("classification", "continuous"),
        expected_rows=("count", "null_count"),
        notes=(
            "classification: columns should include <feature>_all and <feature>_<class>.",
            "continuous: columns should be base feature names only.",
            "text and short_text columns should be excluded.",
            "null_count must be <= count per column.",
        ),
    ),
    StatisticContract(
        statistic_cls=ValueCountsStatistic,
        targets=("classification", "continuous"),
        expected_rows=("value_counts",),
        notes=(
            "categorical columns store list[(value, count)], others are None.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=MeanStatistic,
        targets=("classification", "continuous"),
        expected_rows=("mean",),
        notes=(
            "numeric columns only; others should be None.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=ModeStatistic,
        targets=("classification",),
        expected_rows=("mode",),
        notes=(
            "mode is a list of values and can contain more than one value.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=BoundStatistic,
        targets=("classification",),
        expected_rows=("min", "max"),
        notes=(
            "numeric columns only; others should be None.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=RangeStatistic,
        targets=("classification",),
        expected_rows=("range",),
        notes=(
            "range equals max - min for numeric columns.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=VarianceStatistic,
        targets=("classification",),
        expected_rows=("variance",),
        notes=(
            "numeric columns only; others should be None.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=StdevStatistic,
        targets=("classification",),
        expected_rows=("stdev",),
        notes=(
            "numeric columns only; others should be None.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=KurtosisStatistic,
        targets=("classification",),
        expected_rows=("kurtosis",),
        notes=(
            "numeric columns only; others should be None.",
            "classification: one column per class plus _all.",
            "module exists but is not imported by default.",
        ),
    ),
    StatisticContract(
        statistic_cls=SkewnessStatistic,
        targets=("classification",),
        expected_rows=("skewness",),
        notes=(
            "numeric columns only; others should be None.",
            "classification: one column per class plus _all.",
            "module exists but is not imported by default.",
        ),
    ),
    StatisticContract(
        statistic_cls=QuantileStatistic,
        targets=("classification",),
        expected_rows=(
            "quantile_0.1",
            "quantile_0.25",
            "quantile_0.5",
            "quantile_0.75",
            "quantile_0.9",
        ),
        notes=(
            "numeric columns only; others should be None.",
            "classification: one column per class plus _all.",
        ),
    ),
    StatisticContract(
        statistic_cls=ViolinStatistic,
        targets=("continuous",),
        expected_rows=("violin",),
        notes=(
            "categorical columns produce density/support/quartiles dicts.",
            "numeric columns should be None.",
        ),
    ),
)


PLOT_CONTRACTS: tuple[PlotContract, ...] = (
    PlotContract(
        plot_cls=BarPlot,
        targets=("classification", "continuous"),
        required_rows=("value_counts", "mean"),
        notes=(
            "bar plot should render categorical (value_counts) or numeric (mean) stats.",
            "image should be non-empty bytes.",
        ),
    ),
    PlotContract(
        plot_cls=LinePlot,
        targets=("classification", "continuous"),
        required_rows=("count", "null_count"),
        notes=(
            "line plot should render count vs null_count.",
            "image should be non-empty bytes.",
        ),
    ),
)


def build_statistics_dataframe(dataset: Dataset) -> pd.DataFrame:
    """Build the descriptive statistics dataframe for contracts."""
    computed = pd.DataFrame()
    for statistic_cls in Statistic.all_subclasses():
        statistic = statistic_cls()
        if statistic.suitable(dataset):
            result = statistic.compute(dataset)
            if result is not None and not result.empty:
                computed = pd.concat([computed, result])
    return computed
