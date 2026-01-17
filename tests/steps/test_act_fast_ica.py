"""Tests for ActFastICA step."""
import sys
import types
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import FastICA as _FastICA
except Exception as exc:
    _SKLEARN_AVAILABLE = False
    _SKLEARN_ERROR = exc
else:
    _SKLEARN_AVAILABLE = True
    _SKLEARN_ERROR = None

if _SKLEARN_AVAILABLE:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_PATH = PROJECT_ROOT / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.append(str(SRC_PATH))

    def _ensure_pkg(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        sys.modules[name] = pkg

    # Avoid importing package __init__ side effects for unrelated steps.
    _ensure_pkg("iaml", SRC_PATH / "iaml")
    _ensure_pkg("iaml.actionables", SRC_PATH / "iaml" / "actionables")
    _ensure_pkg(
        "iaml.actionables.features_preprocessing",
        SRC_PATH / "iaml" / "actionables" / "features_preprocessing",
    )

    _candidate_stubbed = False
    if "iaml.candidate" not in sys.modules:
        candidate_module = types.ModuleType("iaml.candidate")

        class Candidate:
            pass

        candidate_module.Candidate = Candidate
        sys.modules["iaml.candidate"] = candidate_module
        _candidate_stubbed = True

    from .step_test_case import StepTestCase
    from iaml.actionables.features_preprocessing.act_fast_ica import ActFastICA

    if _candidate_stubbed:
        sys.modules.pop("iaml.candidate", None)
else:
    ActFastICA = None
    StepTestCase = unittest.TestCase


@unittest.skipUnless(
    _SKLEARN_AVAILABLE,
    f"scikit-learn unavailable: {_SKLEARN_ERROR!r}",
)
class TestActFastICA(StepTestCase):
    def _make_numeric_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
                "f3": [0.5, 1.5, 2.5, 3.5, 2.0, 1.0],
            }
        )

    def test_transform_reduces_to_configured_components(self) -> None:
        df = self._make_numeric_df()
        step = ActFastICA()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.shape, (len(df), 2))
        self.assertEqual(list(result.columns), ["ica_0", "ica_1"])
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_fit_clamps_components_to_data_rank(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [2.0, 0.0, 2.0, 0.0],
                "f3": [0.0, 1.0, 0.0, 1.0],
            }
        )
        step = ActFastICA()
        step.configure("n_components", 10)

        dataset = self.make_dataset(df)
        self.fit_step(step, dataset)

        self.assertEqual(step.get_config("n_components"), 3)
        self.assertEqual(step.component_names, ["ica_0", "ica_1", "ica_2"])

        result = step.transform(df.copy())
        self.assertEqual(result.shape, (len(df), 3))
        self.assertEqual(list(result.columns), ["ica_0", "ica_1", "ica_2"])

    def test_transform_noop_with_single_sample(self) -> None:
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        step = ActFastICA()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
