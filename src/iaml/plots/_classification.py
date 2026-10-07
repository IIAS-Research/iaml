"""Align binary performance curves with the fitted class order."""
from __future__ import annotations

import numpy as np


def binary_score_inputs(estimator, X, y):
    """Return binary targets and scores for ``estimator.classes_[1]``.

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

    if hasattr(estimator, 'decision_function'):
        scores = np.asarray(estimator.decision_function(X))
        if scores.shape != (len(targets),):
            raise ValueError("Binary decision_function must return one score per observation")
    else:
        probabilities = np.asarray(estimator.predict_proba(X))
        if probabilities.shape != (len(targets), 2):
            raise ValueError("Binary predict_proba must return one column for each fitted class")
        scores = probabilities[:, 1]
    return targets == classes[1], scores
