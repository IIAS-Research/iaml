"""Tests for ActPermutationImportanceSelector."""
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
    from iaml.actionables.features_selection.act_permutation_importance_selector import (
        ActPermutationImportanceSelector,
    )
else:
    StepTestCase = unittest.TestCase
    ActPermutationImportanceSelector = None


class TestActPermutationImportanceSelector(StepTestCase):
    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_selects_top_feature_with_max_features(self) -> None:
        n_samples = 30
        signal = list(range(n_samples))
        df = pd.DataFrame(
            {
                "signal": signal,
                "constant": [1] * n_samples,
            }
        )
        y = [value * 2 + 1 for value in signal]
        step = ActPermutationImportanceSelector()
        step.configure("estimator", "linear")
        step.configure("threshold", "none")
        step.configure("max_features", 1)
        step.configure("n_repeats", 3)
        step.configure("random_state", 0)

        result = self.apply_transform(step, df, y)

        expected = df[["signal"]].copy()
        self.assertFrameEqual(result, expected)
        self.assertEqual(step.selected_columns, ["signal"])
        self.assertEqual(step.columns_to_drop, ["constant"])
        self.assertIsNotNone(step.estimator)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_clamps_max_features_and_keeps_all_columns(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6],
                "b": [2, 1, 2, 1, 2, 1],
            }
        )
        y = [1, 2, 3, 4, 5, 6]
        step = ActPermutationImportanceSelector()
        step.configure("estimator", "linear")
        step.configure("threshold", "none")
        step.configure("max_features", 10)
        step.configure("n_repeats", 2)
        step.configure("random_state", 0)

        dataset = self.make_dataset(df, y)
        self.fit_step(step, dataset)

        self.assertEqual(step.get_config("max_features"), 2)
        self.assertCountEqual(step.selected_columns, ["a", "b"])
        self.assertEqual(step.columns_to_drop, [])

        result = step.transform(df.copy())
        self.assertFrameEqual(result, df)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_suitable_requires_threshold_or_max_features(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [5, 4, 3, 2, 1],
            }
        )
        y = [1, 2, 3, 4, 5]
        step = ActPermutationImportanceSelector()
        step.configure("threshold", "none")

        self.assertFalse(step.suitable(self.make_dataset(df, y)))
