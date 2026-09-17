"""Dataset must recognize pandas dtypes without converting the data."""
import unittest

import pandas as pd

from iaml.data_type import DataType
from iaml.dataset import Dataset


class TestDatasetTypes(unittest.TestCase):
    def assert_detected(self, frame, expected):
        original = frame.copy(deep=True)
        dataset = Dataset(frame)
        self.assertEqual(
            dataset.columns_types,
            {name: (frame[name].dtype, kind) for name, kind in expected.items()},
        )
        pd.testing.assert_frame_equal(dataset.X, original)
        pd.testing.assert_frame_equal(frame, original)
        return dataset

    def test_pandas_dtypes_with_missing_values_and_empty_columns(self):
        cases = (
            ("category", ["a", "b", "a"], DataType.CATEGORICAL),
            (pd.CategoricalDtype([1, 2, 3], ordered=True), [1, 2, 1], DataType.CATEGORICAL),
            ("string", ["a", "b", "a"], DataType.CATEGORICAL),
            ("Int64", [1, 2, 3], DataType.NUMERIC),
            ("Float64", [0.1, 0.2, 0.3], DataType.NUMERIC),
            ("boolean", [True, False, True], DataType.CATEGORICAL),
            ("datetime64[ns, UTC]", ["2025-01-01", "2025-01-02", "2025-01-03"],
             DataType.DATE),
        )
        for dtype, values, expected in cases:
            for data in (values, [*values[:-1], None], [None] * 3, []):
                with self.subTest(dtype=dtype, data=data):
                    frame = pd.DataFrame({"feature": pd.Series(data, dtype=dtype)})
                    self.assert_detected(frame, {"feature": expected})

    def test_native_types_and_string_heuristics(self):
        frame = pd.DataFrame({
            "integer": range(8),
            "float": [value / 10 for value in range(8)],
            "bool": [True, False] * 4,
            "date": pd.date_range("2025-01-01", periods=8),
            "duration": pd.to_timedelta(range(8), unit="D"),
        })
        expected = {"integer": DataType.NUMERIC, "float": DataType.NUMERIC,
                    "bool": DataType.CATEGORICAL, "date": DataType.DATE,
                    "duration": DataType.NUMERIC}
        for dtype in ("object", "string"):
            for name, values, kind in (
                ("category", ["a", "b"] * 4, DataType.CATEGORICAL),
                ("short", [f"value {index}" for index in range(8)], DataType.SHORT_TEXT),
                ("long", ["x" * 86 + str(index) for index in range(8)], DataType.TEXT),
            ):
                column = f"{dtype}_{name}"
                frame[column] = pd.Series(values, dtype=dtype)
                expected[column] = kind
        self.assert_detected(frame, expected)

    def test_explicit_categories_stay_categorical_regardless_of_values(self):
        for values in (range(8), ["x" * 86 + str(index) for index in range(8)]):
            with self.subTest(values=values):
                frame = pd.DataFrame({"feature": pd.Series(values, dtype="category")})
                self.assert_detected(frame, {"feature": DataType.CATEGORICAL})

    def test_explicit_column_types_keep_priority(self):
        frame = pd.DataFrame({"code": pd.Series([1, 2, None], dtype="Int64")})
        dataset = Dataset(frame, columns_types={"code": DataType.CATEGORICAL})
        self.assertEqual(dataset.columns_types["code"], (frame.code.dtype, DataType.CATEGORICAL))
        pd.testing.assert_frame_equal(dataset.X, frame)

    def test_transform_redetects_changed_pandas_dtypes(self):
        dataset = Dataset(pd.DataFrame({"value": [1, 2, 3], "category": ["a", "b", "a"]}))
        expected = dataset.X.astype({"value": "Float64", "category": "category"})
        dataset.transform(lambda frame: frame.astype({"value": "Float64", "category": "category"}))
        self.assertEqual(dataset.columns_types, {
            "value": (expected.value.dtype, DataType.NUMERIC),
            "category": (expected.category.dtype, DataType.CATEGORICAL),
        })
        pd.testing.assert_frame_equal(dataset.X, expected)
