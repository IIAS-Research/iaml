"""Lightweight datasets for unit and integration tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _make_base_features(n_samples: int, seed: int) -> tuple[pd.DataFrame, np.random.Generator]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num_1": rng.normal(size=n_samples),
            "num_2": rng.normal(size=n_samples),
            "num_3": rng.normal(size=n_samples),
        }
    )
    return X, rng


def make_regression_data(n_samples: int = 120, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    X, rng = _make_base_features(n_samples, seed)
    noise = rng.normal(scale=0.1, size=n_samples)
    y = 1.5 * X["num_1"] - 2.0 * X["num_2"] + 0.5 * X["num_3"] + noise
    return X, y.to_numpy()


def make_classification_data(n_samples: int = 120, seed: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    X, _ = _make_base_features(n_samples, seed)
    logits = 1.2 * X["num_1"] - 1.0 * X["num_2"] + 0.3 * X["num_3"]
    threshold = np.median(logits)
    y = (logits > threshold).astype(int)
    return X, y.to_numpy()


def make_survival_data(n_samples: int = 120, seed: int = 2) -> tuple[pd.DataFrame, list[tuple[bool, float]]]:
    X, rng = _make_base_features(n_samples, seed)
    event = rng.random(n_samples) > 0.3
    time = rng.exponential(scale=10.0, size=n_samples)
    y = list(zip(event.astype(bool), time.astype(float)))
    return X, y
