"""Tests for ActLogTransformer step."""
import sys
import types
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

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
from iaml.actionables.features_preprocessing.act_log_transformer import ActLogTransformer

if _candidate_stubbed:
    sys.modules.pop("iaml.candidate", None)


class TestActLogTransformer(StepTestCase):
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "skewed": [0.0] * 9 + [1000.0],
                "uniform": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "city": ["a"] * 10,
            }
        )

    def test_transform_applies_log1p_to_skewed_column(self) -> None:
        df = self._make_df()
        step = ActLogTransformer()

        result = self.apply_transform(step, df)

        expected = df.copy()
        expected["skewed"] = np.log1p(expected["skewed"])
        self.assertFrameEqual(result, expected, atol=1e-8, rtol=1e-8)

    def test_min_value_filters_out_columns(self) -> None:
        df = self._make_df()
        step = ActLogTransformer()
        step.configure("min_value", 1.0)

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)

    def test_transform_clips_new_data_at_min_value(self) -> None:
        train_df = self._make_df()
        step = ActLogTransformer()
        dataset = self.make_dataset(train_df)
        self.fit_step(step, dataset)

        new_df = pd.DataFrame(
            {
                "skewed": [-5.0, 0.0, 10.0],
                "uniform": [1.0, 2.0, 3.0],
                "city": ["a", "b", "c"],
            }
        )

        result = step.transform(new_df.copy())

        expected = new_df.copy()
        expected["skewed"] = np.log1p(expected["skewed"].clip(lower=0.0))
        self.assertFrameEqual(result, expected, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
