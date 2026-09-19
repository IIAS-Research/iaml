"""Tests for ActKBinsDiscretizer."""
import sys
from pathlib import Path
import types
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _load_step():
    # Avoid importing iaml/__init__.py and actionables/__init__.py during tests.
    stubs = {}

    def _stub_package(name: str, path: Path) -> None:
        stubs[name] = sys.modules.get(name)
        if stubs[name] is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    _stub_package("iaml", SRC_PATH / "iaml")
    _stub_package("iaml.actionables", SRC_PATH / "iaml" / "actionables")
    _stub_package(
        "iaml.actionables.features_preprocessing",
        SRC_PATH / "iaml" / "actionables" / "features_preprocessing",
    )

    candidate_prev = sys.modules.get("iaml.candidate")
    if candidate_prev is None:
        candidate_module = types.ModuleType("iaml.candidate")

        class Candidate:
            pass

        candidate_module.Candidate = Candidate
        sys.modules["iaml.candidate"] = candidate_module

    try:
        from .step_test_case import StepTestCase
        from iaml.actionables.features_preprocessing.act_k_bins_discretizer import (
            ActKBinsDiscretizer,
        )

        return StepTestCase, ActKBinsDiscretizer
    finally:
        for name, module in stubs.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if candidate_prev is None:
            sys.modules.pop("iaml.candidate", None)


SKIP_REASON = None
try:
    StepTestCase, ActKBinsDiscretizer = _load_step()
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("iaml"):
        raise
    SKIP_REASON = f"Missing optional dependency: {exc.name}"
    StepTestCase = unittest.TestCase
    ActKBinsDiscretizer = None


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON)
class TestActKBinsDiscretizer(StepTestCase):
    def test_transform_ordinal_discretizes_active_columns(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "constant": [7, 7, 7, 7, 7, 7],
            }
        )
        step = ActKBinsDiscretizer()
        step.configure({"n_bins": 3, "encode": "ordinal", "strategy": "uniform"})

        result = self.apply_transform(step, df)

        self.assertListEqual(result.columns.tolist(), df.columns.tolist())
        for col in ["f1", "f2"]:
            self.assertGreaterEqual(result[col].min(), 0.0)
            self.assertLessEqual(result[col].max(), 2.0)
        self.assertTrue((result["constant"] == 7).all())

    def test_transform_onehot_expands_features(self) -> None:
        df = pd.DataFrame(
            {
                "city": ["a", "b", "c", "d"],
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [3.0, 2.0, 1.0, 0.0],
            }
        )
        step = ActKBinsDiscretizer()
        step.configure({"n_bins": 3, "encode": "onehot", "strategy": "uniform"})

        result = self.apply_transform(step, df)

        self.assertEqual(len(result), len(df))
        self.assertIn("city", result.columns)
        self.assertNotIn("x", result.columns)
        self.assertNotIn("y", result.columns)
        self.assertGreater(result.shape[1], df.shape[1])

        encoded = result.drop(columns=["city"])
        self.assertGreater(encoded.shape[1], 0)
        self.assertGreaterEqual(encoded.min().min(), 0.0)
        self.assertLessEqual(encoded.max().max(), 1.0)
        self.assertTrue((encoded.sum(axis=1) == 2).all())

    def test_transform_no_active_columns_returns_input(self) -> None:
        df = pd.DataFrame({"f1": [1.0, 1.0, 1.0], "f2": [2.0, 2.0, 2.0]})
        step = ActKBinsDiscretizer()

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
