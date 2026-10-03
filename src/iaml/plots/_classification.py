"""Align binary performance curves with the fitted probability columns."""
from __future__ import annotations

import numpy as np


def binary_probability_inputs(estimator, X, y):
    """Return binary targets and probabilities for ``estimator.classes_[1]``.

    The fitted estimator defines the class order, even when a test cohort is
    missing one of its classes. Target values are positional, as in ``Dataset``.
    """
    classes = np.asarray(getattr(estimator, "classes_", None))
    if classes.ndim != 1 or len(classes) != 2:
        raise ValueError(
            "Binary performance curves require a fitted estimator with exactly two classes"
        )

    targets = np.asarray(y)
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if targets.ndim != 1:
        raise ValueError("Binary performance curves require a single target column")
    if len(targets) != len(X):
        raise ValueError("Targets must have one value per observation")
    if not len(targets):
        raise ValueError("Binary performance curves require at least one observation")
    if not np.isin(targets, classes).all():
        raise ValueError("Evaluation targets contain labels absent from the fitted classes")

    probabilities = np.asarray(estimator.predict_proba(X))
    if probabilities.shape != (len(targets), 2):
        raise ValueError("Binary predict_proba must return one column for each fitted class")
    return targets == classes[1], probabilities[:, 1]
