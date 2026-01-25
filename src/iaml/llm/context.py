"""Builds LLM-ready context from datasets, statistics, and pipeline stages."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data_type import DataType
from ..dataset import Dataset
from ..logger import Logger
from ..metastep import MetaStep
from ..meta_explorer_step import MetaExplorerStep
from ..meta_ordered_step import MetaOrderedStep
from ..step import Step
from ..void_step import VoidStep
from .settings import LLMSettings
from .utils import safe_serialize


def _is_llm_step_class(step_cls: type[Step]) -> bool:
    if step_cls is VoidStep:
        return False
    if issubclass(step_cls, MetaStep):
        return False
    tags = Step.available_steps.get(step_cls, ())
    if "disabled" in tags:
        return False
    return True


def _serialize_statistics(stats_df: pd.DataFrame, settings: LLMSettings) -> dict[str, Any]:
    if stats_df is None or stats_df.empty:
        return {}

    df = stats_df.copy()
    truncated_columns = False
    truncated_rows = False

    excluded = set(settings.statistics_exclude or ())
    if excluded:
        index_names = df.index.map(str)
        df = df.loc[~index_names.isin(excluded)]
        if df.empty:
            return {}

    if df.shape[1] > settings.max_statistics_columns:
        df = df.iloc[:, : settings.max_statistics_columns]
        truncated_columns = True
    if df.shape[0] > settings.max_statistics_rows:
        df = df.iloc[: settings.max_statistics_rows, :]
        truncated_rows = True

    data = []
    for _, row in df.iterrows():
        data.append(
            [
                safe_serialize(
                    val,
                    settings.max_value_counts,
                    settings.max_statistics_dict_items,
                )
                for val in row.tolist()
            ]
        )

    return {
        "rows": [str(idx) for idx in df.index.tolist()],
        "columns": [str(col) for col in df.columns.tolist()],
        "data": data,
        "truncated": {
            "columns": truncated_columns,
            "rows": truncated_rows,
        },
    }


def _target_summary(dataset: Dataset) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": dataset.type_of_target}
    if dataset.y is None or dataset.type_of_target is None:
        return summary

    y = dataset.y
    if dataset.type_of_target in ["binary", "multiclass"]:
        labels, counts = np.unique(y, return_counts=True)
        total = int(counts.sum())
        summary["class_counts"] = {
            str(label): int(count) for label, count in zip(labels, counts)
        }
        summary["class_ratios"] = {
            str(label): float(count / total) for label, count in zip(labels, counts)
        }
    elif dataset.type_of_target == "continuous":
        summary.update(
            {
                "min": float(np.min(y)),
                "max": float(np.max(y)),
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
            }
        )
    elif dataset.type_of_target == "survival":
        try:
            events = [bool(row["event"]) for row in y]
            times = [float(row["time"]) for row in y]
            summary.update(
                {
                    "event_rate": float(sum(events) / max(1, len(events))),
                    "time_min": float(np.min(times)) if times else None,
                    "time_max": float(np.max(times)) if times else None,
                    "time_mean": float(np.mean(times)) if times else None,
                }
            )
        except Exception:
            pass
    return summary


def _columns_profile(dataset: Dataset) -> dict[str, int]:
    return {
        "numeric": len(dataset.get_columns_names_by_type(DataType.NUMERIC)),
        "categorical": len(dataset.get_columns_names_by_type(DataType.CATEGORICAL)),
        "text": len(
            dataset.get_columns_names_by_type([DataType.TEXT, DataType.SHORT_TEXT])
        ),
        "date": len(dataset.get_columns_names_by_type(DataType.DATE)),
        "missing_columns": int(
            sum(bool(dataset.X[col].isna().any()) for col in dataset.X.columns)
        ),
    }


def _columns_summary(
    dataset: Dataset,
    settings: LLMSettings,
    compact: bool,
) -> list[dict[str, Any]]:
    rows = []
    filtered = []
    for col in dataset.X.columns:
        series = dataset.X[col]
        dtype = str(series.dtype)
        data_type = dataset.columns_types.get(col, (None, None))[1]
        data_type_name = getattr(data_type, "name", None)
        missing_ratio = float(series.isna().mean()) if len(series) else 0.0
        row = {
            "name": str(col),
            "dtype": dtype,
            "data_type": data_type_name,
            "missing_ratio": missing_ratio,
            "unique_count": int(series.nunique(dropna=True)),
        }
        rows.append(row)
        if not compact:
            continue
        is_non_numeric = data_type in (
            DataType.CATEGORICAL,
            DataType.TEXT,
            DataType.SHORT_TEXT,
            DataType.DATE,
        )
        if missing_ratio > 0.0 or is_non_numeric:
            filtered.append(row)

    if not compact:
        return rows

    candidates = filtered or rows
    if candidates:
        candidates.sort(
            key=lambda item: (
                item["missing_ratio"],
                item.get("data_type") != "NUMERIC",
            ),
            reverse=True,
        )
    return candidates[: settings.max_columns_summary]


def build_step_catalog(
    dataset: Dataset,
    allowed_steps: set[str] | None = None,
    include_long_description: bool = False,
    compact: bool = False,
) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for step_cls, tags in Step.available_steps.items():
        if allowed_steps is not None and step_cls.__name__ not in allowed_steps:
            continue
        if not _is_llm_step_class(step_cls):
            continue
        try:
            step = step_cls()
        except Exception as exc:  # noqa: BLE001
            Logger().warning(f"Skipping step {step_cls.__name__}: init failed ({exc}).")
            continue

        suitable = False
        try:
            suitable = step.suitable(dataset)
        except Exception:  # noqa: BLE001
            suitable = False

        config_schema = {}
        for key, conf in step.configuration.items():
            entry = {
                "default": safe_serialize(conf.get("default")),
            }
            range_value = conf.get("range")
            if isinstance(range_value, (list, tuple)):
                if not any(item is not None for item in range_value):
                    range_value = None
            if range_value is not None:
                entry["range"] = safe_serialize(range_value)

            categorical = conf.get("categorical")
            if isinstance(categorical, (list, tuple)) and not categorical:
                categorical = None
            if categorical is not None:
                entry["categorical"] = safe_serialize(categorical)
            if not compact:
                entry["value"] = safe_serialize(conf.get("value"))
                entry["description"] = str(conf.get("description", "")).replace("\n", " ")
            config_schema[key] = entry

        catalog_entry: dict[str, Any] = {
            "config_schema": config_schema,
        }

        if not compact:
            catalog_entry.update(
                {
                    "tags": sorted(list(step.tags or tags)),
                    "usage": getattr(step, "_usage", ""),
                    "description": step.description,
                    "description_long": step.description_long
                    if include_long_description
                    else "",
                    "optimizable": bool(step.optimizable),
                    "can_be_disabled": bool(step.can_be_disabled),
                    "suitable": suitable,
                }
            )

        catalog[step_cls.__name__] = catalog_entry
    return catalog


def build_stage_plan(
    root_step: Step,
    settings: LLMSettings,
    dataset: Dataset | None = None,
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    estimator_tag = None
    if dataset is not None:
        try:
            estimator_tag = dataset.needed_estimator
        except Exception:
            estimator_tag = None
    if isinstance(root_step, MetaOrderedStep):
        children = root_step.steps
    elif hasattr(root_step, "steps"):
        children = root_step.steps
    else:
        children = []

    for child in children:
        tag = getattr(child, "tag", None)
        stage_id = tag or child.name or child.__class__.__name__
        also_explore_without = bool(getattr(child, "also_explore_without", False))
        stage_is_predictor = bool(tag and "predictor" in str(tag))

        if tag:
            allowed = []
            for step_cls, tags in Step.available_steps.items():
                if tag not in tags or not _is_llm_step_class(step_cls):
                    continue
                if estimator_tag and stage_is_predictor and estimator_tag not in tags:
                    continue
                allowed.append(step_cls.__name__)
        else:
            allowed = [
                step.__class__.__name__
                for step in getattr(child, "steps", [])
                if _is_llm_step_class(step.__class__)
            ]
        allowed = sorted(set(allowed))

        stage_type = child.__class__.__name__
        is_explorer = isinstance(child, MetaExplorerStep)

        if stage_is_predictor:
            min_steps = 1
            max_steps = 1
        elif is_explorer:
            min_steps = 0 if also_explore_without else 1
            max_steps = 1
        else:
            min_steps = 0
            max_steps = min(settings.max_steps_per_stage, max(1, len(allowed)))

        stages.append(
            {
                "id": stage_id,
                "name": child.name,
                "tag": tag,
                "stage_type": stage_type,
                "also_explore_without": also_explore_without,
                "allowed_steps": allowed,
                "constraints": {
                    "min_steps": min_steps,
                    "max_steps": max_steps,
                },
            }
        )
    return stages


def build_llm_context(
    dataset: Dataset,
    stats_df: pd.DataFrame,
    root_step: Step,
    settings: LLMSettings,
) -> dict[str, Any]:
    stages = build_stage_plan(root_step, settings, dataset)
    allowed = set()
    for stage in stages:
        allowed.update(stage.get("allowed_steps") or [])

    compact = settings.compact_context
    dataset_summary = {
        "rows": int(dataset.X.shape[0]),
        "columns": int(dataset.X.shape[1]),
        "target_summary": _target_summary(dataset),
        "columns_summary": _columns_summary(dataset, settings, compact),
    }
    if compact:
        dataset_summary["columns_profile"] = _columns_profile(dataset)

    statistics = {}
    if not compact or settings.include_statistics:
        statistics = _serialize_statistics(stats_df, settings)

    return {
        "dataset": dataset_summary,
        "statistics": statistics,
        "stages": stages,
        "steps": build_step_catalog(
            dataset,
            allowed_steps=allowed,
            include_long_description=settings.include_step_description_long,
            compact=compact,
        ),
    }
