"""Tests for ActSelectFromModel."""
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

try:
    import sklearn  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest("scikit-learn is required") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"
ACTIONABLES_PATH = IAML_PATH / "actionables"
FEATURES_SELECTION_PATH = ACTIONABLES_PATH / "features_selection"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


_ensure_package("iaml", IAML_PATH)
_ensure_package("iaml.actionables", ACTIONABLES_PATH)
_ensure_package("iaml.actionables.features_selection", FEATURES_SELECTION_PATH)

from .step_test_case import StepTestCase
from iaml.actionables.features_selection.act_select_from_model import (
    ActSelectFromModel,
)


class TestActSelectFromModel(StepTestCase):
    def test_tree_regressor_drops_constant_numeric(self) -> None:
        n_samples = 20
        y = list(range(n_samples))
        df = pd.DataFrame(
            {
                "signal": y,
                "constant": [1] * n_samples,
                "text": ["a", "b"] * (n_samples // 2),
            }
        )
        step = ActSelectFromModel()
        step.configure("estimator", "tree")
        step.configure("tree_n_estimators", 20)
        step.configure("random_state", 0)

        result = self.apply_transform(step, df, y=y)

        expected = df[["signal", "text"]].copy()
        self.assertFrameEqual(result, expected)
        self.assertIn("constant", step.columns_to_drop)
        self.assertIn("signal", step.selected_columns)

    def test_invalid_threshold_keeps_all_columns(self) -> None:
        df = pd.DataFrame(
            {
                "a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            }
        )
        y = list(range(len(df)))
        step = ActSelectFromModel()
        step.configure("threshold", "not-a-number")

        result = self.apply_transform(step, df, y=y)

        expected = df.copy()
        self.assertFrameEqual(result, expected)
        self.assertEqual(step.columns_to_drop, [])
