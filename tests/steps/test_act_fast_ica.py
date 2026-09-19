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
        for dtype in ("float32", "float64", "int64", "Float64"):
            for whiten in ("unit-variance", "arbitrary-variance"):
                with self.subTest(dtype=dtype, whiten=whiten):
                    values = df.astype(dtype)
                    original = values.copy(deep=True)
                    step = ActFastICA()
                    step.configure("n_components", 10)
                    step.configure("whiten", whiten)
                    dataset = self.make_dataset(values)
                    dataset_original = dataset.X.copy(deep=True)

                    self.fit_step(step, dataset)
                    result = step.transform(values)

                    self.assertEqual(step.get_config("n_components"), 2)
                    self.assertEqual(step.component_names, ["ica_0", "ica_1"])
                    self.assertEqual(result.shape, (len(values), 2))
                    self.assertEqual(list(result.columns), ["ica_0", "ica_1"])
                    self.assertTrue(np.isfinite(result.to_numpy()).all())
                    self.assertFrameEqual(values, original)
                    self.assertFrameEqual(dataset.X, dataset_original)

    def test_fit_keeps_all_components_for_full_rank_data(self) -> None:
        df = pd.DataFrame({
            "f1": [-1.0] * 4 + [1.0] * 4,
            "f2": [-1.0, -1.0, 1.0, 1.0] * 2,
            "f3": [-1.0, 1.0] * 4,
        })
        # Mix the independent signals to avoid a degenerate singular-value basis.
        df["f2"] += 0.2 * df["f1"]
        df["f3"] += 0.3 * df["f1"] + 0.4 * df["f2"]
        step = ActFastICA()
        step.configure("n_components", 10)

        result = self.apply_transform(step, df)

        self.assertEqual(step.get_config("n_components"), 3)
        self.assertEqual(list(result.columns), ["ica_0", "ica_1", "ica_2"])
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_constant_column_does_not_add_a_component(self) -> None:
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 0.0, 2.0, 0.0],
            "constant": [7.0] * 4,
        })
        step = ActFastICA()

        result = self.apply_transform(step, df)

        self.assertEqual(result.shape, (len(df), 2))
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_transform_noop_with_fewer_than_two_independent_features(self) -> None:
        for f1 in ([1.0] * 4, [1.0, 2.0, 3.0, 4.0]):
            with self.subTest(f1=f1):
                df = pd.DataFrame({"f1": f1, "f2": [2 * x + 1 for x in f1]})
                step = ActFastICA()

                result = self.apply_transform(step, df)

                self.assertIsNone(step.preprocessor)
                self.assertFrameEqual(result, df)

    def test_without_whitening_keeps_uncentered_components(self) -> None:
        df = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [2.0, 0.0, 2.0, 0.0],
            "f3": [0.0, 1.0, 0.0, 1.0],
        })
        step = ActFastICA()
        step.configure("whiten", False)
        with self.assertWarnsRegex(UserWarning, "Ignoring n_components"):
            result = self.apply_transform(step, df)

        self.assertEqual(result.shape, (len(df), 3))
        self.assertTrue(np.isfinite(result.to_numpy()).all())

    def test_transform_noop_with_single_sample(self) -> None:
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        step = ActFastICA()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
