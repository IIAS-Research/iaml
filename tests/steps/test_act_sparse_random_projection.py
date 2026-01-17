"""Tests for ActSparseRandomProjection step."""
import importlib.machinery
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

try:
    from sklearn.random_projection import SparseRandomProjection as _SparseRandomProjection
except Exception as exc:
    _IMPORT_ERROR = exc
    StepTestCase = unittest.TestCase
    ActSparseRandomProjection = None
else:
    _IMPORT_ERROR = None

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    SRC_PATH = PROJECT_ROOT / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.append(str(SRC_PATH))

    def _ensure_package(name: str, path: Path) -> None:
        if name in sys.modules:
            return
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        module.__spec__ = importlib.machinery.ModuleSpec(
            name,
            loader=None,
            is_package=True,
        )
        sys.modules[name] = module

    # Avoid importing iaml.__init__ side effects for unrelated steps.
    _ensure_package("iaml", SRC_PATH / "iaml")
    _ensure_package("iaml.actionables", SRC_PATH / "iaml" / "actionables")
    _ensure_package(
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

    try:
        from .step_test_case import StepTestCase
        from iaml.actionables.features_preprocessing.act_sparse_random_projection import (
            ActSparseRandomProjection,
        )
    except ModuleNotFoundError as exc:
        _IMPORT_ERROR = exc
        StepTestCase = unittest.TestCase
        ActSparseRandomProjection = None
    finally:
        if _candidate_stubbed:
            sys.modules.pop("iaml.candidate", None)

_SKIP_REASON = "optional dependency unavailable"
if _IMPORT_ERROR is not None:
    _SKIP_REASON = f"optional dependency unavailable: {_IMPORT_ERROR!r}"


@unittest.skipIf(ActSparseRandomProjection is None, _SKIP_REASON)
class TestActSparseRandomProjection(StepTestCase):
    def test_transform_reduces_numeric_columns_and_preserves_text(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d"],
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [4.0, 3.0, 2.0, 1.0],
                "f3": [0.5, 1.5, 2.5, 3.5],
            }
        )
        step = ActSparseRandomProjection()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertEqual(result.shape, (len(df), 3))
        self.assertEqual(list(result.columns), ["text", "srp_0", "srp_1"])
        self.assertEqual(result["text"].tolist(), df["text"].tolist())
        self.assertTrue(np.isfinite(result[["srp_0", "srp_1"]].to_numpy()).all())

    def test_fit_clamps_components_to_max_allowed(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0],
                "f2": [2.0, 1.0, 0.0],
            }
        )
        step = ActSparseRandomProjection()
        step.configure("n_components", 10)

        dataset = self.make_dataset(df)
        self.fit_step(step, dataset)

        self.assertEqual(step.get_config("n_components"), 2)
        self.assertEqual(step.component_names, ["srp_0", "srp_1"])

        result = step.transform(df.copy())
        self.assertEqual(result.shape, (len(df), 2))
        self.assertEqual(list(result.columns), ["srp_0", "srp_1"])

    def test_transform_noop_with_single_sample(self) -> None:
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        step = ActSparseRandomProjection()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
