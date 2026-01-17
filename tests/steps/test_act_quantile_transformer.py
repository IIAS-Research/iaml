"""Tests for ActQuantileTransformer."""
import sys
import unittest
from enum import Enum
from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

ACT_PATH = SRC_PATH / "iaml" / "actionables" / "features_preprocessing" / "act_quantile_transformer.py"


def _load_act_quantile_transformer_from_source():
    try:
        from sklearn.preprocessing import QuantileTransformer
    except ModuleNotFoundError:
        return None

    import textwrap

    class Step:
        available_steps: dict = {}

        def __init__(self, *args, **kwargs):
            self.configuration = {}
            self.tags = set()

        def default_configuration(self) -> None:
            for param in self.configuration.values():
                param["value"] = param.get("value", param.get("default"))

        def configure(self, key, value=None) -> None:
            if isinstance(key, dict):
                for name, val in key.items():
                    self.configure(name, val)
                return
            if key not in self.configuration:
                raise AttributeError(f"Configurable Key '{key}' does not exist.")
            self.configuration[key]["value"] = value

        def passthrough_parameters(self, default: bool = True) -> dict:
            parameters = {}
            for key, value in self.configuration.items():
                if "passthrough" in value:
                    if value["passthrough"]:
                        parameters[key] = value.get("value", value.get("default"))
                elif default:
                    parameters[key] = value.get("value", value.get("default"))
            return parameters

    class Actionable(Step):
        pass

    class Dataset:
        pass

    class Candidate:
        pass

    class DataType(Enum):
        NUMERIC = 3

    def is_step(*tags):
        def wrapper(cls):
            Step.available_steps[cls] = tags
            initial_init = cls.__init__

            def __init__(self, *args, **kwargs):
                Step.__init__(self)
                self.tags = set(tags)
                initial_init(self, *args, **kwargs)
                self.default_configuration()

            cls.__init__ = __init__
            return cls

        return wrapper

    source = ACT_PATH.read_text(encoding="utf-8")
    filtered_lines = []
    for line in source.splitlines():
        if line.startswith("import ") or line.startswith("from "):
            continue
        filtered_lines.append(line)

    module_globals = {
        "__file__": str(ACT_PATH),
        "__name__": "_iaml_act_quantile_transformer",
        "textwrap": textwrap,
        "pd": pd,
        "QuantileTransformer": QuantileTransformer,
        "Actionable": Actionable,
        "Dataset": Dataset,
        "Candidate": Candidate,
        "DataType": DataType,
        "is_step": is_step,
    }
    exec("\n".join(filtered_lines), module_globals)
    return module_globals["ActQuantileTransformer"]


def _load_act_quantile_transformer():
    try:
        from iaml.actionables.features_preprocessing.act_quantile_transformer import (
            ActQuantileTransformer,
        )
        return ActQuantileTransformer
    except ModuleNotFoundError as exc:
        if exc.name == "sklearn":
            return None
    except Exception:
        pass
    return _load_act_quantile_transformer_from_source()


ActQuantileTransformer = _load_act_quantile_transformer()


class _MiniDataset:
    def __init__(self, X: pd.DataFrame):
        self.X = X

    def get_columns_names_by_type(self, _types):
        return [col for col in self.X.columns if is_numeric_dtype(self.X[col])]


def _apply_transform(step, df: pd.DataFrame) -> pd.DataFrame:
    dataset = _MiniDataset(df)
    step.fit(dataset)
    return step.transform(df.copy())


class TestActQuantileTransformer(unittest.TestCase):
    @unittest.skipIf(ActQuantileTransformer is None, "scikit-learn is required")
    def test_transform_uniform_on_numeric_columns(self) -> None:
        df = pd.DataFrame(
            {
                "age": [10, 20, 30, 40],
                "score": [1.0, 3.0, 2.0, 5.0],
                "city": ["paris", "lyon", "nice", "dijon"],
            }
        )
        step = ActQuantileTransformer()
        step.configure("output_distribution", "uniform")

        result = _apply_transform(step, df)

        self.assertListEqual(result.columns.tolist(), df.columns.tolist())
        numeric = result[["age", "score"]]
        self.assertGreaterEqual(numeric.min().min(), 0.0)
        self.assertLessEqual(numeric.max().max(), 1.0)
        self.assertListEqual(result["city"].tolist(), df["city"].tolist())
        self.assertNotEqual(numeric.max().max(), df[["age", "score"]].max().max())

    @unittest.skipIf(ActQuantileTransformer is None, "scikit-learn is required")
    def test_no_numeric_columns_returns_input(self) -> None:
        df = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        step = ActQuantileTransformer()

        result = _apply_transform(step, df)

        pd.testing.assert_frame_equal(result, df)

    @unittest.skipIf(ActQuantileTransformer is None, "scikit-learn is required")
    def test_empty_dataframe_returns_input(self) -> None:
        df = pd.DataFrame(
            {
                "age": pd.Series(dtype="int64"),
                "score": pd.Series(dtype="float64"),
            }
        )
        step = ActQuantileTransformer()

        result = _apply_transform(step, df)

        pd.testing.assert_frame_equal(result, df)
