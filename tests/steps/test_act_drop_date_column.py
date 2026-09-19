"""Tests for ActDropDateColumn."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.cleaning.act_drop_date_column import ActDropDateColumn
from iaml.data_type import DataType
from iaml.dataset import Dataset


def make_dataset_with_types(df: pd.DataFrame, types: dict[str, DataType]) -> Dataset:
    missing = set(df.columns) - set(types)
    if missing:
        raise ValueError(f"Missing types for columns: {sorted(missing)}")
    columns_types = {name: (df[name].dtype, dtype) for name, dtype in types.items()}
    return Dataset(df, y=[0] * len(df), columns_types=columns_types)


class TestActDropDateColumn(StepTestCase):
    def test_fit_drops_date_columns_and_sets_explanations(self) -> None:
        df = pd.DataFrame(
            {
                "created_at": pd.to_datetime(["2021-01-01", "2021-01-02"]),
                "updated_at": pd.to_datetime(["2021-02-01", "2021-02-02"]),
                "value": [10, 20],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "created_at": DataType.DATE,
                "updated_at": DataType.DATE,
                "value": DataType.NUMERIC,
            },
        )
        step = ActDropDateColumn()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        expected_dropped = ["created_at", "updated_at"]
        expected_explanations = [
            f"Dropped column **`{column}`**." for column in expected_dropped
        ]
        self.assertCountEqual(step.columns_to_drop, expected_dropped)
        self.assertCountEqual(step.explanations, expected_explanations)
        self.assertEqual(len(step.explanations), len(step.columns_to_drop))

        result = step.transform(dataset.X.copy())

        expected = df.drop(expected_dropped, axis=1)
        self.assertFrameEqual(result, expected)

    def test_no_date_columns_noop_and_not_suitable(self) -> None:
        df = pd.DataFrame(
            {
                "name": ["alice", "bob"],
                "score": [1.5, 2.5],
            }
        )
        dataset = make_dataset_with_types(
            df,
            {
                "name": DataType.SHORT_TEXT,
                "score": DataType.NUMERIC,
            },
        )
        step = ActDropDateColumn()

        self.assertFalse(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertEqual(step.columns_to_drop, [])
        self.assertEqual(step.explanations, [])

        result = step.transform(dataset.X.copy())

        self.assertFrameEqual(result, df)
