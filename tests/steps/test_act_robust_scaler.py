"""Tests for ActRobustScaler."""
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
        from iaml.actionables.normalize.act_robust_scaler import ActRobustScaler

        return StepTestCase, ActRobustScaler
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
    StepTestCase, ActRobustScaler = _load_step()
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("iaml"):
        raise
    SKIP_REASON = f"Missing optional dependency: {exc.name}"
    StepTestCase = unittest.TestCase
    ActRobustScaler = None


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON)
class TestActRobustScaler(StepTestCase):
    def test_scales_numeric_columns_only(self) -> None:
        df = pd.DataFrame(
            {
                "age": [0, 1, 2],
                "score": [10.0, 20.0, 30.0],
                "city": ["paris", "lyon", "nice"],
            }
        )
        step = ActRobustScaler()

        result = self.apply_transform(step, df)

        numeric = df[["age", "score"]]
        median = numeric.median()
        q1 = numeric.quantile(0.25)
        q3 = numeric.quantile(0.75)
        iqr = q3 - q1
        scaled = (numeric - median) / iqr
        expected = pd.DataFrame(
            {
                "age": scaled["age"],
                "score": scaled["score"],
                "city": df["city"],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_transform_uses_fitted_iqr(self) -> None:
        train = pd.DataFrame(
            {
                "age": [0, 10],
                "score": [1.0, 3.0],
                "city": ["a", "b"],
            }
        )
        step = ActRobustScaler()
        dataset = self.make_dataset(train)
        self.fit_step(step, dataset)

        new = pd.DataFrame(
            {
                "age": [5, 10],
                "score": [2.0, 3.0],
                "city": ["c", "d"],
            }
        )
        result = step.transform(new.copy())

        expected = pd.DataFrame(
            {
                "age": [0.0, 1.0],
                "score": [0.0, 1.0],
                "city": ["c", "d"],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_no_numeric_columns_returns_input(self) -> None:
        df = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        step = ActRobustScaler()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        self.assertFrameEqual(result, expected)
