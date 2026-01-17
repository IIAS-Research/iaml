"""Tests for ActRFE."""
from pathlib import Path
import sys
from types import ModuleType
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _ensure_stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


# Avoid importing iaml.__init__ (it pulls in broken optional steps).
_ensure_stub_package("iaml", SRC_PATH / "iaml")
_ensure_stub_package("iaml.actionables", SRC_PATH / "iaml" / "actionables")
_ensure_stub_package(
    "iaml.actionables.features_selection",
    SRC_PATH / "iaml" / "actionables" / "features_selection",
)

if not hasattr(sys.modules["iaml"], "actionables"):
    sys.modules["iaml"].actionables = sys.modules["iaml.actionables"]
if not hasattr(sys.modules["iaml.actionables"], "features_selection"):
    sys.modules["iaml.actionables"].features_selection = sys.modules[
        "iaml.actionables.features_selection"
    ]

try:
    import sklearn  # noqa: F401
except ImportError:
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True

if SKLEARN_AVAILABLE:
    from .step_test_case import StepTestCase
    from iaml.actionables.features_selection.act_rfe import ActRFE
else:
    StepTestCase = unittest.TestCase
    ActRFE = None


class TestActRFE(StepTestCase):
    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_selects_signal_and_drops_constant(self) -> None:
        df = pd.DataFrame(
            {
                "signal": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "constant": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        step = ActRFE()
        step.configure("n_features_to_select", 1)
        step.configure("step", 1)

        result = self.apply_transform(step, df, y)

        expected = df[["signal"]]
        self.assertFrameEqual(result, expected)
        self.assertEqual(step.selected_columns, ["signal"])
        self.assertEqual(step.columns_to_drop, ["constant"])
        self.assertIsNotNone(step.selector)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_fit_keeps_all_features_when_selecting_too_many(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [3.0, 2.0, 1.0],
            }
        )
        y = [0.0, 1.0, 2.0]
        step = ActRFE()
        step.configure("n_features_to_select", 5)

        dataset = self.make_dataset(df, y)
        self.fit_step(step, dataset)

        self.assertEqual(step.selected_columns, ["a", "b"])
        self.assertEqual(step.columns_to_drop, [])
        self.assertIsNone(step.selector)

        result = step.transform(df.copy())
        self.assertFrameEqual(result, df)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_suitable_rejects_survival_target(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
            }
        )
        y = [(1, 5), (0, 6), (1, 7)]
        step = ActRFE()
        step.configure("n_features_to_select", 1)

        self.assertFalse(step.suitable(self.make_dataset(df, y)))
