"""Tests for ActNormalizer."""
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
    _stub_package("iaml.actionables.normalize", SRC_PATH / "iaml" / "actionables" / "normalize")

    candidate_prev = sys.modules.get("iaml.candidate")
    if candidate_prev is None:
        candidate_module = types.ModuleType("iaml.candidate")

        class Candidate:
            pass

        candidate_module.Candidate = Candidate
        sys.modules["iaml.candidate"] = candidate_module

    try:
        from .step_test_case import StepTestCase
        from iaml.actionables.normalize.act_normalizer import ActNormalizer

        return StepTestCase, ActNormalizer
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
    StepTestCase, ActNormalizer = _load_step()
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("iaml"):
        raise
    SKIP_REASON = f"Missing optional dependency: {exc.name}"
    StepTestCase = unittest.TestCase
    ActNormalizer = None


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON)
class TestActNormalizer(StepTestCase):
    def test_l2_normalizes_numeric_columns_only(self) -> None:
        df = pd.DataFrame(
            {
                "age": [3.0, 0.0],
                "score": [4.0, 0.0],
                "city": ["paris", "lyon"],
            }
        )
        step = ActNormalizer()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "age": [0.6, 0.0],
                "score": [0.8, 0.0],
                "city": ["paris", "lyon"],
            }
        )
        self.assertFrameEqual(result, expected, atol=1e-6, rtol=1e-6)

    def test_l1_normalization_config(self) -> None:
        df = pd.DataFrame(
            {
                "x": [1.0, -1.0],
                "y": [2.0, 1.0],
                "city": ["a", "b"],
            }
        )
        step = ActNormalizer()
        step.configure("norm", "l1")

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "x": [1.0 / 3.0, -0.5],
                "y": [2.0 / 3.0, 0.5],
                "city": ["a", "b"],
            }
        )
        self.assertFrameEqual(result, expected, atol=1e-6, rtol=1e-6)

    def test_no_numeric_columns_returns_input(self) -> None:
        df = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        step = ActNormalizer()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        self.assertFrameEqual(result, expected)
