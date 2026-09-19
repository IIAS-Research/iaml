"""Tests for ActBorderlineSMOTE."""
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
    from imblearn.over_sampling import BorderlineSMOTE
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False
    BorderlineSMOTE = None

if IMBLEARN_AVAILABLE:
    from iaml.actionables.imbalance.act_borderline_smote import ActBorderlineSMOTE


@unittest.skipUnless(IMBLEARN_AVAILABLE, "imblearn is required for BorderlineSMOTE tests")
class TestActBorderlineSMOTE(StepTestCase):
    def test_resample_balances_minority_class(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "feature_b": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            }
        )
        y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        step = ActBorderlineSMOTE()
        step.configure({"k_neighbors": 1, "m_neighbors": 3})

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        counts = pd.Series(y_resampled).value_counts().to_dict()
        self.assertEqual(counts[0], 8)
        self.assertEqual(counts[1], 8)
        self.assertEqual(len(X_resampled), 16)
        self.assertListEqual(list(X_resampled.columns), ["feature_a", "feature_b"])

    def test_resample_noop_with_single_minority(self) -> None:
        df = pd.DataFrame(
            {
                "value": [1, 2, 3, 4, 5],
                "score": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        y = [0, 0, 0, 0, 1]
        step = ActBorderlineSMOTE()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        self.assertFrameEqual(X_resampled, df)
        self.assertListEqual(list(y_resampled), y)
        self.assertIsNone(step.resampler)

    def test_effective_neighbors_are_clamped(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "feature_b": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            }
        )
        y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        step = ActBorderlineSMOTE()

        dataset = self.make_dataset(df, y)
        step.fit(dataset)

        self.assertEqual(step._effective_k_neighbors, 1)
        self.assertEqual(step._effective_m_neighbors, 9)
        self.assertIsInstance(step.resampler, BorderlineSMOTE)
