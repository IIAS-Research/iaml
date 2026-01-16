"""Base helpers for step tests."""
from __future__ import annotations

from typing import Iterable
import sys
from pathlib import Path
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

# Ensure src/ is importable when running tests from the repo.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from iaml.dataset import Dataset


class StepTestCase(unittest.TestCase):
    """Shared helpers to make writing step tests concise."""

    def make_dataset(self, X: pd.DataFrame, y: Iterable | None = None) -> Dataset:
        if y is None:
            y = [0] * len(X)
        return Dataset(X, y)

    def fit_step(self, step: object, dataset: Dataset) -> object:
        try:
            return step.fit(dataset)
        except TypeError:
            return step.fit(dataset.X, dataset.y)

    def apply_transform(self, step: object, X: pd.DataFrame, y: Iterable | None = None) -> pd.DataFrame:
        dataset = self.make_dataset(X, y)
        self.fit_step(step, dataset)
        if not hasattr(step, "transform"):
            raise AttributeError("Step has no transform method")
        return step.transform(dataset.X.copy())

    def apply_resample(self, step: object, X: pd.DataFrame, y: Iterable) -> tuple[pd.DataFrame, Iterable]:
        dataset = self.make_dataset(X, y)
        self.fit_step(step, dataset)
        if not hasattr(step, "resample"):
            raise AttributeError("Step has no resample method")
        X_resampled, y_resampled = step.resample(dataset.X.copy(), dataset.y)
        return X_resampled, y_resampled

    def assertFrameEqual(self, left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> None:
        try:
            assert_frame_equal(left, right, **kwargs)
        except AssertionError as exc:
            self.fail(str(exc))
