"""Tests for ActTruncatedSVD step."""
import importlib.machinery
from pathlib import Path
import sys
import types
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    sys.modules[name] = module


# Avoid importing iaml/__init__.py when loading submodules in tests.
_ensure_package("iaml", SRC_PATH / "iaml")
_ensure_package("iaml.actionables", SRC_PATH / "iaml" / "actionables")
_ensure_package(
    "iaml.actionables.features_preprocessing",
    SRC_PATH / "iaml" / "actionables" / "features_preprocessing",
)

try:
    from .step_test_case import StepTestCase
    from iaml.actionables.features_preprocessing.act_truncated_svd import ActTruncatedSVD
except ModuleNotFoundError as exc:
    if exc.name in {"sklearn", "rich", "multiprocess"}:
        StepTestCase = unittest.TestCase
        ActTruncatedSVD = None
    else:
        raise


class TestActTruncatedSVD(StepTestCase):
    @unittest.skipIf(ActTruncatedSVD is None, "scikit-learn is required")
    def test_transform_reduces_numeric_columns_and_preserves_non_numeric(self) -> None:
        df = pd.DataFrame(
            {
                "text": ["a", "b", "c", "d", "e"],
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "f2": [2.0, 1.0, 0.0, 1.0, 2.0],
                "f3": [0.0, 1.0, 0.0, 1.0, 0.0],
            }
        )
        step = ActTruncatedSVD()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertEqual(result.shape, (len(df), 3))
        self.assertEqual(list(result.columns), ["text", "svd_0", "svd_1"])
        self.assertEqual(result["text"].tolist(), df["text"].tolist())
        self.assertTrue(np.isfinite(result[["svd_0", "svd_1"]].to_numpy()).all())

    @unittest.skipIf(ActTruncatedSVD is None, "scikit-learn is required")
    def test_fit_clamps_components_to_max_rank(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0],
                "f2": [2.0, 0.0, 2.0, 0.0],
                "f3": [0.0, 1.0, 0.0, 1.0],
            }
        )
        step = ActTruncatedSVD()
        step.configure("n_components", 10)

        dataset = self.make_dataset(df)
        self.fit_step(step, dataset)

        self.assertEqual(step.get_config("n_components"), 2)
        self.assertEqual(step.component_names, ["svd_0", "svd_1"])

        result = step.transform(df.copy())
        self.assertEqual(result.shape, (len(df), 2))
        self.assertEqual(list(result.columns), ["svd_0", "svd_1"])

    @unittest.skipIf(ActTruncatedSVD is None, "scikit-learn is required")
    def test_transform_noop_with_insufficient_samples(self) -> None:
        df = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
        step = ActTruncatedSVD()
        step.configure("n_components", 2)

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
