"""Tests for ActPoissonRegressor step."""
import importlib
import sys
import types
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import PoissonRegressor
except ImportError:  # pragma: no cover - optional dependency
    PoissonRegressor = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"
ALIAS_PACKAGE = "iaml_test"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


def _ensure_module(name: str, attrs: dict) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _DatasetStub:
    def __init__(self, X: pd.DataFrame, y, type_of_target: str = "continuous"):
        self.X = X
        self.y = np.array(y)
        self.type_of_target = type_of_target

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    def get_columns_names_by_type(self, _data_type) -> list[str]:
        return list(self.X.select_dtypes(include=[np.number]).columns)


class _CandidateStub:
    pass


# Import the step under an alias package to avoid full-package side effects.
for package, path in [
    (ALIAS_PACKAGE, IAML_PATH),
    (f"{ALIAS_PACKAGE}.actionables", IAML_PATH / "actionables"),
    (f"{ALIAS_PACKAGE}.actionables.predictors", IAML_PATH / "actionables" / "predictors"),
    (
        f"{ALIAS_PACKAGE}.actionables.predictors.regressor",
        IAML_PATH / "actionables" / "predictors" / "regressor",
    ),
]:
    _ensure_package(package, path)

_ensure_module(f"{ALIAS_PACKAGE}.dataset", {"Dataset": _DatasetStub})
_ensure_module(f"{ALIAS_PACKAGE}.candidate", {"Candidate": _CandidateStub})

try:
    ActPoissonRegressor = importlib.import_module(
        f"{ALIAS_PACKAGE}.actionables.predictors.regressor.act_poisson_regressor"
    ).ActPoissonRegressor
except ImportError:  # pragma: no cover - optional dependency
    ActPoissonRegressor = None

_SKIP_REASON = None
if PoissonRegressor is None:
    _SKIP_REASON = "scikit-learn is not available"
elif ActPoissonRegressor is None:
    _SKIP_REASON = "ActPoissonRegressor could not be imported"


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON)
class TestActPoissonRegressor(unittest.TestCase):
    def _make_features(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "f1": [0.0, 1.0, 2.0, 3.0],
                "f2": [1.0, 0.5, 1.5, 2.0],
            }
        )

    def _make_dataset(self, X: pd.DataFrame, y) -> _DatasetStub:
        return _DatasetStub(X, y, type_of_target="continuous")

    def _fit_step(self, step, dataset):
        try:
            return step.fit(dataset)
        except TypeError:
            return step.fit(dataset.X, dataset.y)

    def test_fit_sets_model_and_predicts_positive(self) -> None:
        df = self._make_features()
        y = [1, 2, 4, 3]
        step = ActPoissonRegressor()
        dataset = self._make_dataset(df, y)

        result = self._fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, PoissonRegressor)

        predictions = step.predict(df)
        self.assertEqual(predictions.shape, (len(df),))
        self.assertTrue(np.isfinite(predictions).all())
        self.assertTrue((predictions > 0).all())

        score = step.score(df, y)
        self.assertTrue(np.isfinite(score))

    def test_suitable_detects_count_targets(self) -> None:
        df = self._make_features()
        step = ActPoissonRegressor()

        counts = self._make_dataset(df, y=[0, 1, 2, 3])
        non_integer = self._make_dataset(df, y=[0.5, 1.2, 2.0, 3.1])
        negative = self._make_dataset(df, y=[0, -1, 2, 3])

        self.assertTrue(step.suitable(counts))
        self.assertFalse(step.suitable(non_integer))
        self.assertFalse(step.suitable(negative))

    def test_fit_uses_numeric_columns_only(self) -> None:
        df = pd.DataFrame(
            {
                "num": [0.0, 1.0, 2.0, 3.0],
                "cat": ["a", "b", "a", "b"],
            }
        )
        y = [1, 2, 3, 4]
        step = ActPoissonRegressor()
        dataset = self._make_dataset(df, y)

        self._fit_step(step, dataset)

        self.assertEqual(step.columns, ["num"])
        predictions = step.predict(df)
        self.assertEqual(predictions.shape, (len(df),))
