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


def make_mixed_feature_data(n_samples: int = 120, seed: int = 10) -> pd.DataFrame:
    X, rng = _make_base_features(n_samples, seed)
    X["cat_small"] = rng.choice(["A", "B", "C"], size=n_samples)
    X["cat_with_nan"] = rng.choice(
        np.array(["yes", "no", None], dtype=object),
        size=n_samples,
        p=[0.45, 0.45, 0.10]
    )
    X["short_text"] = [f"id_{i:04d}" for i in range(n_samples)]
    long_base = (
        "lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "sed do eiusmod tempor incididunt."
    )
    X["long_text"] = [f"{long_base} {i}" for i in range(n_samples)]
    X["date_col"] = (
        pd.to_datetime("2020-01-01")
        + pd.to_timedelta(np.arange(n_samples), unit="D")
    )

    nan_idx = rng.choice(n_samples, size=max(1, n_samples // 10), replace=False)
    X.loc[nan_idx, "num_2"] = np.nan
    return X


def make_statistics_classification_data(
    n_samples: int = 120,
    seed: int = 20,
    n_classes: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    X = make_mixed_feature_data(n_samples, seed)
    rng = np.random.default_rng(seed + 1)
    labels = np.tile(np.arange(n_classes), int(np.ceil(n_samples / n_classes)))[:n_samples]
    rng.shuffle(labels)
    return X, labels


def make_statistics_regression_data(
    n_samples: int = 120,
    seed: int = 30) -> tuple[pd.DataFrame, np.ndarray]:
    X = make_mixed_feature_data(n_samples, seed)
    rng = np.random.default_rng(seed + 2)
    num_2 = X["num_2"].fillna(X["num_2"].mean())
    noise = rng.normal(scale=0.1, size=n_samples)
    y = 1.5 * X["num_1"] - 2.0 * num_2 + 0.5 * X["num_3"] + noise
    return X, y.to_numpy()


def make_statistics_survival_data(
    n_samples: int = 120,
    seed: int = 40) -> tuple[pd.DataFrame, list[tuple[bool, float]]]:
    X = make_mixed_feature_data(n_samples, seed)
    rng = np.random.default_rng(seed + 3)
    event = rng.random(n_samples) > 0.3
    time = rng.exponential(scale=10.0, size=n_samples)
    y = list(zip(event.astype(bool), time.astype(float)))
    return X, y
