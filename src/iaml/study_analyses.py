"""Compile study collections and compute independent analytical outputs.

These helpers contain no search operations. Their definitions capture result
keys and configuration before components have a chance to mutate during use.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .cache_keys import hash_evaluation_context
from .logger import Logger
from .metric import Metric
from .metric_plot import MetricPlot


@dataclass
class CompiledAnalysis:
    """An independent component with its fixed output key and input signature."""

    key: str
    component: Any
    configuration: dict[str, Any]
    signature: str | None
    alias: str | None = None
    kind: str = ""


def _definition(component, alias=None, kind="") -> CompiledAnalysis:
    configuration = deepcopy(vars(component))
    key = alias or str(component)
    # Plot has no stable __str__; its public class name is its canonical key.
    if alias is None and isinstance(component, MetricPlot):
        key = type(component).__name__
    signature = hash_evaluation_context(type(component), configuration, key, kind)
    return CompiledAnalysis(key, component, configuration, signature, alias, kind)


def compile_analyses(collection) -> list[CompiledAnalysis]:
    """Freeze the local definitions of a declarative analysis collection."""
    definitions = [
        _definition(spec.instantiate(), spec.alias, collection.kind)
        for spec in collection.resolved()
    ]
    _check_result_keys(definitions, collection.kind)
    return definitions


def _check_result_keys(definitions, kind):
    keys = [definition.key for definition in definitions]
    duplicate = next((key for key in keys if keys.count(key) > 1), None)
    if duplicate is not None:
        raise ValueError(
            f"Several {kind} produce '{duplicate}'; name each variant with .named(...)"
        )


def analyses_signature(definitions) -> str | None:
    """Return a stable pre-computation signature, or decline caching."""
    if any(definition.signature is None for definition in definitions):
        return None
    return hash_evaluation_context(tuple(definition.signature for definition in definitions))


def compile_metrics(collection, main_metric, dataset):
    """Return ``(definitions, objective_key, applicability_report)``.

    An historical objective instance retains its exact configuration, provided
    its class occurs once. Excluded or ambiguous objectives fail before search.
    """
    specs = collection.resolved()
    if isinstance(main_metric, Metric):
        matching = [spec for spec in specs if spec.component is type(main_metric)]
        if len(matching) != 1:
            raise ValueError(
                "The main metric is excluded or ambiguous; select a declared metric alias"
            )
        selected = matching[0]
        # Do not construct the objective class: its supplied instance is the
        # authoritative definition, including any required constructor inputs.
        definitions = [
            _definition(deepcopy(main_metric) if spec is selected else spec.instantiate(),
                        spec.alias, "metrics")
            for spec in specs
        ]
        objective = next(definition.key for spec, definition in zip(specs, definitions)
                         if spec is selected)
    else:
        definitions = [_definition(spec.instantiate(), spec.alias, "metrics") for spec in specs]
    _check_result_keys(definitions, "metrics")
    if isinstance(main_metric, str):
        objective = main_metric
    elif main_metric is None:
        objective = {
            "classifier": "balanced_accuracy",
            "survival": "concordance_index_ipcw",
            "regressor": "r2_score",
        }[dataset.needed_estimator]
        # A single alias on the default objective still names the same metric.
        matching = [d for d in definitions if str(d.component) == objective]
        if len(matching) == 1:
            objective = matching[0].key
    elif main_metric is not None and not isinstance(main_metric, Metric):
        raise TypeError("main_metric must be a Metric, a result key, or None")

    if objective not in {definition.key for definition in definitions}:
        raise ValueError(f"Main metric '{objective}' is absent from the metrics collection")

    applicable, report = [], []
    for definition in definitions:
        suitable = definition.component.suitable(dataset.X, dataset.y, dataset.type_of_target)
        if suitable or (main_metric is not None and definition.key == objective):
            applicable.append(definition)
            if not suitable:
                # Existing suitable() methods sometimes express an automatic
                # recommendation (e.g. accuracy on unbalanced classes). An
                # explicit objective keeps its exact definition; fold scoring
                # still rejects unusable predictions or invalid metric outputs.
                report.append({"key": definition.key, "status": "selected",
                               "reason": "Explicit objective bypasses the automatic suitability filter"})
        else:
            record = {"key": definition.key, "status": "inapplicable", "reason":
                      f"Not suitable for target '{dataset.type_of_target}'"}
            report.append(record)
            if definition.key == objective:
                raise ValueError(f"Main metric '{objective}' is incompatible with this dataset")
    return applicable, objective, report


def compute_statistics(definitions, dataset):
    """Return an aliased DataFrame and a report, preserving canonical plot rows."""
    tables, canonical_tables, report = [], [], []
    for definition in deepcopy(definitions):
        component = definition.component
        try:
            if not component.suitable(dataset):
                report.append({"key": definition.key, "status": "inapplicable",
                               "reason": f"Not suitable for target '{dataset.type_of_target}'"})
                continue
            result = component.compute(deepcopy(dataset))
            if result is None or result.empty:
                report.append({"key": definition.key, "status": "inapplicable",
                               "reason": "No applicable columns or outputs"})
                continue
            canonical = result.copy(deep=True)
            canonical_tables.append(canonical)
            display = result.copy(deep=True)
            if definition.alias:
                display.index = (
                    [definition.alias] if len(display.index) == 1
                    else [f"{definition.alias} : {label}" for label in display.index]
                )
            tables.append(display)
            report.append({"key": definition.key, "status": "success"})
        except Exception as exc:  # independent analyses must retain successful outputs
            Logger().warning(f"Statistic '{definition.key}' failed: {exc!r}")
            report.append({"key": definition.key, "status": "error", "reason": repr(exc)})
    table = pd.concat(tables) if tables else pd.DataFrame()
    if table.index.has_duplicates:
        duplicate = table.index[table.index.duplicated()][0]
        raise ValueError(f"Several statistics produce row '{duplicate}'; name each variant with .named(...)")
    canonical = pd.concat(canonical_tables) if canonical_tables else pd.DataFrame()
    # attrs carry the plotting representation without altering visible labels.
    # Store plain records rather than a nested DataFrame (pandas attrs equality).
    table.attrs["iaml_canonical_rows"] = list(canonical.index)
    table.attrs["iaml_statistics_signature"] = analyses_signature(definitions)
    table.attrs["iaml_dataset_fingerprint"] = dataset.fingerprint()
    return table, report


def canonical_statistics(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy suitable for the existing canonical-label plot adapters."""
    result = copy_statistics(table)
    labels = table.attrs.get("iaml_canonical_rows")
    if labels is not None and len(labels) == len(result.index):
        result.index = labels
    result.attrs = {}
    # Multiple variants may share a plot's canonical rows. The first declared
    # variant supplies the existing plot adapter, while the table keeps them all.
    return result.loc[~result.index.duplicated(keep="first")]


def copy_statistics(table: pd.DataFrame) -> pd.DataFrame:
    """Copy also nested object cells, which pandas' deep copy shares otherwise."""
    result = table.copy(deep=True)
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].map(deepcopy)
    result.attrs = deepcopy(table.attrs)
    return result
