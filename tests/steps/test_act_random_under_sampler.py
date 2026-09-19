"""Tests for ActRandomUnderSampler."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import pandas as pd
from pandas.testing import assert_frame_equal


try:
    from imblearn.under_sampling import RandomUnderSampler  # noqa: F401
    IMBLEARN_AVAILABLE = True
except Exception:
    IMBLEARN_AVAILABLE = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")
    sys.modules[name] = pkg
    return pkg


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_step_and_dataset() -> tuple[type, type]:
    # Avoid importing iaml/__init__.py which pulls in optional modules.
    _ensure_package("iaml", IAML_PATH)
    _ensure_package("iaml.actionables", IAML_PATH / "actionables")
    _ensure_package("iaml.actionables.imbalance", IAML_PATH / "actionables" / "imbalance")

    step_module = _load_module(
        "iaml.actionables.imbalance.act_random_under_sampler",
        IAML_PATH / "actionables" / "imbalance" / "act_random_under_sampler.py",
    )
    dataset_module = sys.modules.get("iaml.dataset")
    if dataset_module is None:
        dataset_module = _load_module("iaml.dataset", IAML_PATH / "dataset.py")
    return step_module.ActRandomUnderSampler, dataset_module.Dataset


if IMBLEARN_AVAILABLE:
    try:
        ActRandomUnderSampler, Dataset = _load_step_and_dataset()
        IMPORT_ERROR = None
    except ImportError as exc:
        ActRandomUnderSampler = None
        Dataset = None
        IMPORT_ERROR = exc
else:
    ActRandomUnderSampler = None
    Dataset = None
    IMPORT_ERROR = None


@unittest.skipUnless(IMBLEARN_AVAILABLE, "imblearn is not available")
class TestActRandomUnderSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if ActRandomUnderSampler is None or Dataset is None:
            raise unittest.SkipTest(
                f"ActRandomUnderSampler import failed: {IMPORT_ERROR!r}"
            )

    def apply_resample(self, step: object, X: pd.DataFrame, y: list) -> tuple[pd.DataFrame, list]:
        dataset = Dataset(X, y)
        step.fit(dataset)
        if not hasattr(step, "resample"):
            raise AttributeError("Step has no resample method")
        X_resampled, y_resampled = step.resample(dataset.X.copy(), dataset.y)
        return X_resampled, y_resampled

    def assertFrameEqual(self, left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> None:
        try:
            assert_frame_equal(left, right, **kwargs)
        except AssertionError as exc:
            self.fail(str(exc))

    def test_resample_balances_majority_class(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4, 5, 6, 7],
                "feature_b": [10, 11, 12, 13, 14, 15, 16],
            }
        )
        y = ["majority", "majority", "majority", "majority", "majority", "minority", "minority"]
        step = ActRandomUnderSampler()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        counts = pd.Series(y_resampled).value_counts().to_dict()
        self.assertEqual(counts["majority"], 2)
        self.assertEqual(counts["minority"], 2)
        self.assertEqual(len(X_resampled), 4)
        self.assertListEqual(list(X_resampled.columns), ["feature_a", "feature_b"])

    def test_resample_noop_when_already_balanced(self) -> None:
        df = pd.DataFrame({"feature": [1, 2, 3, 4]})
        y = [0, 0, 1, 1]
        step = ActRandomUnderSampler()

        X_resampled, y_resampled = self.apply_resample(step, df, y)

        self.assertFrameEqual(X_resampled, df)
        self.assertListEqual(list(y_resampled), y)

    def test_resample_with_string_labels(self) -> None:
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        y = ["no", "no", "no", "yes", "yes"]
        step = ActRandomUnderSampler()

        _, y_resampled = self.apply_resample(step, df, y)

        counts = pd.Series(y_resampled).value_counts().to_dict()
        self.assertEqual(counts["no"], 2)
        self.assertEqual(counts["yes"], 2)
