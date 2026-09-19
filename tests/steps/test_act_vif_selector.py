"""Tests for ActVIFSelector."""
from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from iaml.actionables.features_selection.act_vif_selector import ActVIFSelector
from iaml.dataset import Dataset


class TestActVIFSelector(unittest.TestCase):
    def make_dataset(self, X: pd.DataFrame, y=None) -> Dataset:
        return Dataset(X, y)

    def fit_step(self, step: object, dataset: Dataset) -> object:
        return step.fit(dataset)

    def apply_transform(self, step: object, X: pd.DataFrame, y=None) -> pd.DataFrame:
        dataset = self.make_dataset(X, y)
        self.fit_step(step, dataset)
        if not hasattr(step, "transform"):
            raise AttributeError("Step has no transform method")
        return step.transform(dataset.X.copy())

    def assertFrameEqual(self, left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> None:
        try:
            assert_frame_equal(left, right, **kwargs)
        except AssertionError as exc:
            self.fail(str(exc))

    def test_drops_high_vif_columns(self) -> None:
        df = pd.DataFrame(
            {
                "signal": [1, 2, 3, 4, 5, 6, 7, 8],
                "collinear": [2, 4, 6, 8, 10, 12, 14, 16.1],
                "other": [1, 0, 1, 0, 1, 0, 1, 0],
            }
        )
        dataset = self.make_dataset(df)
        step = ActVIFSelector()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertCountEqual(step.to_drop, ["signal", "collinear"])

        result = step.transform(dataset.X.copy())

        expected = df[["other"]].copy()
        self.assertFrameEqual(result, expected)

    def test_keeps_columns_when_vif_below_threshold(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, 0, 2, 1, 3, 0],
            }
        )
        dataset = self.make_dataset(df)
        step = ActVIFSelector()

        self.assertFalse(step.suitable(dataset))

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)

    def test_constant_column_does_not_trigger_drop(self) -> None:
        df = pd.DataFrame(
            {
                "constant": [1, 1, 1, 1],
                "signal": [1, 2, 3, 4],
            }
        )
        dataset = self.make_dataset(df)
        step = ActVIFSelector()

        self.assertFalse(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.to_drop, [])

        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)
