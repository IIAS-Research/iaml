"""Tests for ActSelectKBest."""
import sys
import unittest

import pandas as pd

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True


def _is_callable_union_error(exc: TypeError) -> bool:
    message = str(exc)
    return (
        "unsupported operand type(s) for |" in message
        and "builtin_function_or_method" in message
    )


def _import_with_callable_patch(importer):
    import builtins
    from typing import Callable

    # Patch builtins so the module's `callable | None` annotation can import.
    class _CallableShim:
        def __init__(self, original_callable):
            self._original_callable = original_callable

        def __call__(self, obj):
            return self._original_callable(obj)

        def __or__(self, other):
            return Callable | other

    original_callable = builtins.callable
    builtins.callable = _CallableShim(original_callable)
    try:
        return importer()
    finally:
        builtins.callable = original_callable


def _import_step_test_case():
    if not SKLEARN_AVAILABLE:
        return None
    try:
        from .step_test_case import StepTestCase
        return StepTestCase
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("sklearn"):
            return None
        raise
    except TypeError as exc:
        if not _is_callable_union_error(exc):
            raise
        module_name = f"{__package__}.step_test_case" if __package__ else "step_test_case"
        sys.modules.pop(module_name, None)
        return _import_with_callable_patch(
            lambda: __import__(module_name, fromlist=["StepTestCase"]).StepTestCase
        )


StepTestCase = _import_step_test_case()
if StepTestCase is None:
    class StepTestCase(unittest.TestCase):
        pass


def _import_act_select_k_best():
    if not SKLEARN_AVAILABLE:
        return None
    try:
        from iaml.actionables.features_selection.act_select_k_best import ActSelectKBest
        return ActSelectKBest
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("sklearn"):
            return None
        raise
    except TypeError as exc:
        if not _is_callable_union_error(exc):
            raise
        module_name = "iaml.actionables.features_selection.act_select_k_best"
        sys.modules.pop(module_name, None)
        return _import_with_callable_patch(
            lambda: __import__(
                module_name, fromlist=["ActSelectKBest"]
            ).ActSelectKBest
        )


ActSelectKBest = _import_act_select_k_best()


class TestActSelectKBest(StepTestCase):
    @unittest.skipIf(ActSelectKBest is None, "scikit-learn is required")
    def test_selects_top_k_features(self) -> None:
        df = pd.DataFrame(
            {
                "signal": [0, 10, 1, 11, 2, 12],
                "noise": [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            }
        )
        y = [0, 1, 0, 1, 0, 1]
        step = ActSelectKBest()
        step.configure("score_func", "f_classif")
        step.configure("k", 1)

        result = self.apply_transform(step, df, y)

        expected = df[["signal"]].copy()
        self.assertFrameEqual(result, expected)
        self.assertEqual(step.selected_columns, ["signal"])
        self.assertEqual(step.columns_to_drop, ["noise"])

    @unittest.skipIf(ActSelectKBest is None, "scikit-learn is required")
    def test_clamps_k_to_feature_count(self) -> None:
        df = pd.DataFrame(
            {
                "a": [0, 1, 0, 1],
                "b": [1, 0, 1, 0],
            }
        )
        y = [0, 1, 0, 1]
        step = ActSelectKBest()
        step.configure("score_func", "f_classif")
        step.configure("k", 10)

        dataset = self.make_dataset(df, y)
        self.fit_step(step, dataset)

        self.assertEqual(step.get_config("k"), 2)
        self.assertEqual(set(step.selected_columns), {"a", "b"})
        self.assertEqual(step.columns_to_drop, [])

        result = step.transform(df.copy())
        self.assertFrameEqual(result, df)

    @unittest.skipIf(ActSelectKBest is None, "scikit-learn is required")
    def test_suitable_rejects_negative_values_for_chi2(self) -> None:
        df = pd.DataFrame({"a": [0, -1, 2], "b": [1, 2, 3]})
        dataset = self.make_dataset(df, y=[0, 1, 0])
        step = ActSelectKBest()
        step.configure("score_func", "chi2")

        self.assertFalse(step.suitable(dataset))
