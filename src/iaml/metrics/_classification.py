"""Shared positive-class convention for binary classification metrics."""
from typing import Any

from sklearn.utils.multiclass import unique_labels


def resolve_pos_label(y: Any, pos_label: Any = None, y_train: Any = None) -> Any:
    """Resolve the positive label without depending on row order or predictions.

    Training labels take precedence when available. For 0/1 and -1/1 targets,
    the positive label is always 1, including all-negative evaluation subsets.
    Otherwise, use the last of two sorted labels. A single nonstandard label
    is ambiguous and requires an explicit positive label or both training classes.
    """
    if pos_label is not None:
        return pos_label

    labels = unique_labels(y_train if y_train is not None else y)
    if 0 < len(labels) <= 2:
        if all(label in (0, 1) for label in labels) or all(label in (-1, 1) for label in labels):
            return 1
        if len(labels) == 2:
            return labels[-1]

    raise ValueError(
        "Cannot infer the positive class: set pos_label explicitly or provide "
        "y_train containing both binary classes."
    )
