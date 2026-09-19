"""Tests for ActNearMiss."""
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
    from imblearn.under_sampling import NearMiss
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    NearMiss = None

if IMBLEARN_AVAILABLE:
    from iaml.actionables.imbalance.act_near_miss import ActNearMiss


@unittest.skipUnless(IMBLEARN_AVAILABLE, "imblearn is required for NearMiss tests")
class TestActNearMiss(StepTestCase):
    def test_resample_reduces_majority_class(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [0.0, 1.0, 2.0, 3.0, 10.0, 10.5],
                "feature_b": [0.0, 1.0, 2.0, 3.0, 10.0, 10.5],
            }
        )
        y = ["majority", "majority", "majority", "majority", "minority", "minority"]
        step = ActNearMiss()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        counts = pd.Series(y_resampled).value_counts().to_dict()
        self.assertEqual(counts["majority"], 2)
        self.assertEqual(counts["minority"], 2)
        self.assertEqual(len(X_resampled), 4)
        self.assertListEqual(list(X_resampled.columns), ["feature_a", "feature_b"])
        self.assertIsInstance(step.resampler, NearMiss)

    def test_resample_noop_when_balanced(self) -> None:
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4],
                "score": [0.1, 0.2, 0.3, 0.4],
            }
        )
        y = [0, 0, 1, 1]
        step = ActNearMiss()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        self.assertFrameEqual(X_resampled, df)
        self.assertListEqual(list(y_resampled), y)
        self.assertIsNone(step.resampler)

    def test_effective_neighbors_are_clamped(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [0, 1, 2, 3, 4],
                "feature_b": [0.0, 0.5, 1.0, 1.5, 2.0],
            }
        )
        y = ["majority", "majority", "majority", "minority", "minority"]
        step = ActNearMiss()
        step.configure("n_neighbors", 10)
        step.configure("n_neighbors_ver3", 10)

        dataset = self.make_dataset(df, y)
        step.fit(dataset)

        self.assertEqual(step._effective_n_neighbors, 2)
        self.assertEqual(step._effective_n_neighbors_ver3, 3)
        self.assertIsInstance(step.resampler, NearMiss)
