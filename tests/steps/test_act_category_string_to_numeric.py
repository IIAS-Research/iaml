"""Tests for ActCategoryStringToNumeric."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_encode_target_column import (
    ActCategoryStringToNumeric,
)


class TestActCategoryStringToNumeric(StepTestCase):
    def assert_category_encoding(self, raw: np.ndarray, encoded: np.ndarray) -> None:
        self.assertEqual(raw.shape, encoded.shape)
        self.assertTrue(np.issubdtype(encoded.dtype, np.integer))
        raw_unique = np.unique(raw)
        encoded_unique = np.unique(encoded)
        self.assertEqual(len(encoded_unique), len(raw_unique))
        for value in raw_unique:
            codes = np.unique(encoded[raw == value])
            self.assertEqual(len(codes), 1)

    def test_fit_sets_column_and_explanations(self) -> None:
        df = pd.DataFrame({"feature": [1, 2, 3]})
        y = ["tea", "coffee", "tea"]
        dataset = self.make_dataset(df, y)
        step = ActCategoryStringToNumeric()

        self.fit_step(step, dataset)

        self.assertEqual(len(step.column_to_encode), 1)
        np.testing.assert_array_equal(step.column_to_encode[0], dataset.y)
        categories = np.unique(dataset.y)
        self.assertEqual(len(step.explanations), len(categories))
        explanation_text = "\n".join(step.explanations)
        for i, value in enumerate(categories):
            self.assertIn(f"`{value}`: {i}", explanation_text)

    def test_transform_encodes_object_categories(self) -> None:
        step = ActCategoryStringToNumeric()
        y = np.array(["coffee", "tea", "coffee", "water"], dtype=object)
        df = pd.DataFrame({"feature": [0] * len(y)})

        self.fit_step(step, self.make_dataset(df, y))

        result = step.transform(y)

        self.assert_category_encoding(y, result)

    def test_transform_encodes_booleans(self) -> None:
        step = ActCategoryStringToNumeric()
        y = np.array([True, False, True], dtype=bool)
        df = pd.DataFrame({"feature": [0] * len(y)})

        self.fit_step(step, self.make_dataset(df, y))

        result = step.transform(y)

        self.assert_category_encoding(y, result)

    def test_transform_passthrough_numeric(self) -> None:
        step = ActCategoryStringToNumeric()
        y = np.array([2, 1, 2], dtype=int)
        df = pd.DataFrame({"feature": [0] * len(y)})

        self.fit_step(step, self.make_dataset(df, y))

        result = step.transform(y)

        np.testing.assert_array_equal(result, y)
