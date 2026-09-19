"""Tests for ActAalenAdditiveFitter step."""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import sys
import types

import numpy as np
import pandas as pd

from .step_test_case import StepTestCase


class AalenAdditiveFitter:
    """Minimal stub for lifelines.AalenAdditiveFitter when lifelines is absent."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_args = None

    def fit(self, df: pd.DataFrame, duration_col: str, event_col: str):
        self.fit_args = {
            "df": df.copy(),
            "duration_col": duration_col,
            "event_col": event_col,
        }
        return self

    def predict_cumulative_hazard(self, X: pd.DataFrame) -> pd.DataFrame:
        values = np.arange(len(X), dtype=float)
        return pd.DataFrame([values], columns=list(X.index))


@contextmanager
def _patched_fitter(step_cls: type):
    globals_dict = step_cls.fit.__globals__
    original = globals_dict.get("AalenAdditiveFitter")
    globals_dict["AalenAdditiveFitter"] = AalenAdditiveFitter
    try:
        yield
    finally:
        if original is None:
            globals_dict.pop("AalenAdditiveFitter", None)
        else:
            globals_dict["AalenAdditiveFitter"] = original


def _import_step() -> type:
    module_name = "iaml.actionables.predictors.survival.act_aalen_additive_model"
    try:
        import lifelines  # noqa: F401
    except ImportError:
        lifelines_stub = types.ModuleType("lifelines")
        lifelines_stub.AalenAdditiveFitter = AalenAdditiveFitter
        sys.modules.pop(module_name, None)
        sys.modules["lifelines"] = lifelines_stub
        module = importlib.import_module(module_name)
        sys.modules.pop("lifelines", None)
        sys.modules.pop(module_name, None)
        return module.ActAalenAdditiveFitter
    module = importlib.import_module(module_name)
    return module.ActAalenAdditiveFitter


ActAalenAdditiveFitter = _import_step()


class TestActAalenAdditiveFitter(StepTestCase):
    def test_adapter_stays_outside_automatic_prediction(self) -> None:
        step = ActAalenAdditiveFitter()
        self.assertEqual(step.tags, {'experimental'})

    def _make_survival_dataset(self) -> tuple[pd.DataFrame, list[tuple[bool, float]]]:
        X = pd.DataFrame(
            {
                "age": [
                    25,
                    30,
                    35,
                    40,
                    45,
                    50,
                    55,
                    60,
                    65,
                    70,
                    28,
                    33,
                    38,
                    43,
                    48,
                    53,
                    58,
                    63,
                    68,
                    73,
                ],
                "marker": [
                    0.8,
                    1.1,
                    0.4,
                    1.5,
                    0.7,
                    1.2,
                    0.5,
                    1.6,
                    0.9,
                    1.3,
                    0.6,
                    1.0,
                    0.3,
                    1.4,
                    0.8,
                    1.1,
                    0.6,
                    1.5,
                    0.7,
                    1.2,
                ],
            }
        )
        events = [
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
        ]
        times = [
            6.0,
            9.5,
            5.5,
            12.0,
            8.2,
            10.5,
            7.1,
            13.4,
            9.0,
            11.8,
            6.7,
            8.9,
            5.9,
            12.5,
            7.8,
            10.9,
            8.4,
            13.0,
            9.6,
            11.2,
        ]
        y = list(zip(events, times))
        return X, y

    def test_fit_sets_model_and_predicts(self) -> None:
        X, y = self._make_survival_dataset()
        dataset = self.make_dataset(X, y)
        with _patched_fitter(ActAalenAdditiveFitter):
            step = ActAalenAdditiveFitter()
            result = self.fit_step(step, dataset)
            predictions = step.predict(X)

        self.assertIs(result, step)
        self.assertIsInstance(step.model, AalenAdditiveFitter)
        self.assertIsInstance(predictions, pd.DataFrame)
        self.assertGreater(predictions.shape[0], 0)
        self.assertGreater(predictions.shape[1], 0)
        self.assertTrue(np.isfinite(predictions.to_numpy()).all())
        self.assertEqual(predictions.shape[1], len(X))
        self.assertListEqual(list(predictions.columns), list(X.index))

        fit_args = step.model.fit_args
        self.assertIsNotNone(fit_args)
        self.assertEqual(fit_args["duration_col"], "time")
        self.assertEqual(fit_args["event_col"], "event")
        fit_df = fit_args["df"]
        self.assertListEqual(list(fit_df.columns), list(X.columns) + ["event", "time"])
        self.assertListEqual(list(fit_df["event"]), [event for event, _ in y])
        self.assertListEqual(list(fit_df["time"]), [time for _, time in y])

    def test_suitable_detects_survival_target(self) -> None:
        X, y = self._make_survival_dataset()
        survival_dataset = self.make_dataset(X, y)
        non_survival_dataset = self.make_dataset(X, [0, 1] * (len(X) // 2))
        step = ActAalenAdditiveFitter()

        self.assertTrue(step.suitable(survival_dataset))
        self.assertFalse(step.suitable(non_survival_dataset))
