"""Tests for ActCyclicalDateEncoding."""
import sys
from pathlib import Path
import types

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _load_step():
    # Avoid importing iaml/__init__.py and actionables/__init__.py during tests.
    stubs = {}

    def _stub_package(name: str, path: Path) -> None:
        stubs[name] = sys.modules.get(name)
        if stubs[name] is None:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module

    _stub_package("iaml", SRC_PATH / "iaml")
    _stub_package("iaml.actionables", SRC_PATH / "iaml" / "actionables")
    _stub_package(
        "iaml.actionables.features_preprocessing",
        SRC_PATH / "iaml" / "actionables" / "features_preprocessing",
    )

    candidate_prev = sys.modules.get("iaml.candidate")
    if candidate_prev is None:
        candidate_module = types.ModuleType("iaml.candidate")

        class Candidate:
            pass

        candidate_module.Candidate = Candidate
        sys.modules["iaml.candidate"] = candidate_module

    try:
        from .step_test_case import StepTestCase
        from iaml.actionables.features_preprocessing.act_cyclical_date_encoding import (
            ActCyclicalDateEncoding,
        )
        from iaml.data_type import DataType
        from iaml.dataset import Dataset

        return StepTestCase, ActCyclicalDateEncoding, DataType, Dataset
    finally:
        for name, module in stubs.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if candidate_prev is None:
            sys.modules.pop("iaml.candidate", None)


StepTestCase, ActCyclicalDateEncoding, DataType, Dataset = _load_step()


def make_dataset_with_types(
    df: pd.DataFrame, types: dict[str, DataType]
) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=[0] * len(df), columns_types=columns_types)


class TestActCyclicalDateEncoding(StepTestCase):
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "created_at": pd.to_datetime(
                    ["2024-01-01 10:00:00", "2024-02-04 22:00:00"]
                ),
                "value": [1, 2],
            }
        )

    def test_transform_encodes_components_and_drops_original(self) -> None:
        df = self._make_df()
        dataset = make_dataset_with_types(
            df,
            {
                "created_at": DataType.DATE,
                "value": DataType.NUMERIC,
            },
        )
        step = ActCyclicalDateEncoding()
        step.configure("components", ("month", "dayofweek"))

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        expected = df.copy()
        month_values = expected["created_at"].dt.month.astype(float)
        dow_values = expected["created_at"].dt.dayofweek.astype(float)
        expected["created_at_month_sin"] = np.sin(
            2.0 * np.pi * month_values / 12.0
        )
        expected["created_at_month_cos"] = np.cos(
            2.0 * np.pi * month_values / 12.0
        )
        expected["created_at_dayofweek_sin"] = np.sin(
            2.0 * np.pi * dow_values / 7.0
        )
        expected["created_at_dayofweek_cos"] = np.cos(
            2.0 * np.pi * dow_values / 7.0
        )
        expected = expected.drop(columns=["created_at"])

        self.assertFrameEqual(result, expected, rtol=1e-8, atol=1e-8)
        self.assertEqual(step.columns, ["created_at"])
        self.assertEqual(len(step.explanations), 1)
        self.assertIn("`created_at`", step.explanations[0])

    def test_transform_keeps_original_when_configured(self) -> None:
        df = self._make_df()
        dataset = make_dataset_with_types(
            df,
            {
                "created_at": DataType.DATE,
                "value": DataType.NUMERIC,
            },
        )
        step = ActCyclicalDateEncoding()
        step.configure("components", "month")
        step.configure("drop_original", False)

        self.fit_step(step, dataset)
        result = step.transform(dataset.X.copy())

        expected = df.copy()
        month_values = expected["created_at"].dt.month.astype(float)
        expected["created_at_month_sin"] = np.sin(
            2.0 * np.pi * month_values / 12.0
        )
        expected["created_at_month_cos"] = np.cos(
            2.0 * np.pi * month_values / 12.0
        )

        self.assertFrameEqual(result, expected, rtol=1e-8, atol=1e-8)
        self.assertIn("created_at", result.columns)

    def test_noop_when_component_is_constant(self) -> None:
        df = pd.DataFrame(
            {
                "created_at": pd.to_datetime(
                    ["2024-01-01 08:00:00", "2024-01-15 18:30:00"]
                ),
                "value": [3, 4],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "created_at": DataType.DATE,
                "value": DataType.NUMERIC,
            },
        )
        step = ActCyclicalDateEncoding()
        step.configure("components", "month")

        self.fit_step(step, dataset)

        self.assertFalse(step.suitable(dataset))
        self.assertEqual(step.columns, [])
        result = step.transform(dataset.X.copy())
        self.assertFrameEqual(result, df)
