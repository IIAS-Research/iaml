"""Tests for ActLightGBMClassifier step."""
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

def _ensure_package(name: str, path: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    elif not hasattr(module, "__path__"):
        module.__path__ = [str(path)]

def _bootstrap_iaml() -> None:
    project_root = Path(__file__).resolve().parents[2]
    iaml_root = project_root / "src" / "iaml"
    _ensure_package("iaml", iaml_root)
    _ensure_package("iaml.actionables", iaml_root / "actionables")
    _ensure_package(
        "iaml.actionables.predictors",
        iaml_root / "actionables" / "predictors",
    )
    _ensure_package(
        "iaml.actionables.predictors.classifier",
        iaml_root / "actionables" / "predictors" / "classifier",
    )

_bootstrap_iaml()
_IMPORT_ERROR = None
try:
    import numpy as np
    import pandas as pd
    from .step_test_case import StepTestCase
    import iaml.actionables.predictors.classifier.act_light_gbm_classifier as lightgbm_module

    ActLightGBMClassifier = lightgbm_module.ActLightGBMClassifier
except ImportError as exc:
    _IMPORT_ERROR = exc
    np = None
    pd = None
    StepTestCase = unittest.TestCase
    ActLightGBMClassifier = None
    lightgbm_module = None

class TestActLightGBMClassifier(StepTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _IMPORT_ERROR is not None:
            raise unittest.SkipTest(f"optional dependency missing: {_IMPORT_ERROR}")
        if lightgbm_module is None or lightgbm_module.LGBMClassifier is None:
            raise unittest.SkipTest("lightgbm is required")

    def _configure_fast(self, step: ActLightGBMClassifier) -> None:
        step.configure(
            {
                "n_estimators": 10,
                "learning_rate": 0.2,
                "num_leaves": 15,
                "max_depth": 3,
                "min_child_samples": 5,
                "random_state": 7,
                "verbosity": -1,
            }
        )

    def _make_features(self, include_text: bool = False) -> pd.DataFrame:
        values = list(range(30))
        data = {
            "age": values,
            "score": [value % 5 for value in values],
            "group": ["a" if value % 2 == 0 else "b" for value in values],
        }
        if include_text:
            data["note"] = [f"note_{value}" for value in values]
        return pd.DataFrame(data)

    def test_fit_predict_proba_and_score_prefers_supported_columns(self) -> None:
        df = self._make_features(include_text=True)
        y = [value % 2 for value in range(len(df))]
        step = ActLightGBMClassifier()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        result = self.fit_step(step, dataset)

        self.assertIs(result, step)
        self.assertIsNotNone(step.model)
        self.assertIsInstance(step.model, lightgbm_module.LGBMClassifier)
        self.assertListEqual(step.columns, ["age", "score", "group"])

        predictions = step.predict(df)
        self.assertEqual(len(predictions), len(df))
        self.assertSetEqual(set(step.classes_), {0, 1})
        self.assertTrue(set(predictions).issubset({0, 1}))

        predictions_supported = step.predict(df[step.columns])
        self.assertListEqual(list(predictions), list(predictions_supported))

        proba = step.predict_proba(df)
        self.assertEqual(proba.shape, (len(df), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))

        score = step.score(df, y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_configuration_passthrough(self) -> None:
        df = self._make_features()
        y = [value % 2 for value in range(len(df))]
        step = ActLightGBMClassifier()
        step.configure(
            {
                "boosting_type": "gbdt",
                "n_estimators": 12,
                "learning_rate": 0.05,
                "num_leaves": 23,
                "max_depth": 4,
                "min_child_samples": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.2,
                "class_weight": "balanced",
                "random_state": 11,
                "verbosity": 0,
            }
        )
        dataset = self.make_dataset(df, y)

        self.fit_step(step, dataset)

        params = step.model.get_params()
        self.assertEqual(params["boosting_type"], "gbdt")
        self.assertEqual(params["n_estimators"], 12)
        self.assertAlmostEqual(params["learning_rate"], 0.05)
        self.assertEqual(params["num_leaves"], 23)
        self.assertEqual(params["max_depth"], 4)
        self.assertEqual(params["min_child_samples"], 6)
        self.assertAlmostEqual(params["subsample"], 0.9)
        self.assertAlmostEqual(params["colsample_bytree"], 0.8)
        self.assertAlmostEqual(params["reg_alpha"], 0.1)
        self.assertAlmostEqual(params["reg_lambda"], 0.2)
        self.assertEqual(params["class_weight"], "balanced")
        self.assertEqual(params["random_state"], 11)
        self.assertEqual(params["verbosity"], 0)

    def test_fit_wraps_lightgbm_error(self) -> None:
        df = self._make_features()
        y = [value % 2 for value in range(len(df))]
        step = ActLightGBMClassifier()
        self._configure_fast(step)
        dataset = self.make_dataset(df, y)

        error_type = lightgbm_module._LGBM_ERRORS[0]
        with patch.object(
            lightgbm_module.LGBMClassifier,
            "fit",
            side_effect=error_type("boom"),
        ):
            with self.assertRaises(ValueError) as ctx:
                self.fit_step(step, dataset)

        self.assertIn("LightGBMClassifier training failed", str(ctx.exception))

    def test_suitable_target_and_supported_columns(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                "f2": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            }
        )
        step = ActLightGBMClassifier()

        binary = self.make_dataset(
            df, y=["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"]
        )
        multiclass = self.make_dataset(
            df, y=["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]
        )
        multilabel = self.make_dataset(
            df,
            y=[
                [True, False],
                [False, True],
                [True, True],
                [False, False],
                [True, False],
                [False, True],
                [True, True],
                [False, False],
                [True, False],
                [False, True],
            ],
        )
        continuous = self.make_dataset(
            df,
            y=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        text_only = self.make_dataset(
            pd.DataFrame({"note": [f"note_{i}" for i in range(10)]}),
            y=["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        )

        self.assertTrue(step.suitable(binary))
        self.assertTrue(step.suitable(multiclass))
        self.assertTrue(step.suitable(multilabel))
        self.assertFalse(step.suitable(continuous))
        self.assertFalse(step.suitable(text_only))
