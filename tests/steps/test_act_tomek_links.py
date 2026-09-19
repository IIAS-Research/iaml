"""Tests for ActTomekLinks."""
import sys
from pathlib import Path
from types import ModuleType
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _ensure_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = ModuleType(name)
    pkg.__path__ = [str(path)]
    sys.modules[name] = pkg


_ensure_pkg("iaml", SRC_PATH / "iaml")
_ensure_pkg("iaml.actionables", SRC_PATH / "iaml" / "actionables")
_ensure_pkg("iaml.actionables.imbalance", SRC_PATH / "iaml" / "actionables" / "imbalance")

from .step_test_case import StepTestCase

try:
    from imblearn.under_sampling import TomekLinks
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    TomekLinks = None

if IMBLEARN_AVAILABLE:
    from iaml.actionables.imbalance.act_tomek_links import ActTomekLinks


@unittest.skipUnless(IMBLEARN_AVAILABLE, "imblearn is required for TomekLinks tests")
class TestActTomekLinks(StepTestCase):
    def test_resample_removes_tomek_majority_sample(self) -> None:
        df = pd.DataFrame(
            {
                "x": [0.0, 1.0, 0.9, 10.0],
                "y": [0.0, 1.0, 0.9, 10.0],
            }
        )
        y = ["majority", "majority", "minority", "majority"]
        step = ActTomekLinks()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        counts = pd.Series(y_resampled).value_counts().to_dict()
        self.assertEqual(counts["majority"], 2)
        self.assertEqual(counts["minority"], 1)
        self.assertEqual(len(X_resampled), 3)
        self.assertListEqual(list(X_resampled.columns), ["x", "y"])
        self.assertIsInstance(step.resampler, TomekLinks)

    def test_resample_noop_with_single_class(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0.1, 0.2, 0.3]})
        y = [1, 1, 1]
        step = ActTomekLinks()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        self.assertFrameEqual(X_resampled, df)
        self.assertListEqual(list(y_resampled), y)
        self.assertIsNone(step.resampler)
