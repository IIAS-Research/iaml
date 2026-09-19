"""Tests for ActDropIdLikeColumns."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_precleaning.act_drop_id_like_columns import (
    ActDropIdLikeColumns,
)


class TestActDropIdLikeColumns(StepTestCase):
    def test_drops_id_like_columns_by_name_and_value(self) -> None:
        rows = 25
        df = pd.DataFrame(
            {
                "user_id": list(range(1000, 1000 + rows)),
                "session": [f"{i:032x}" for i in range(rows)],
                "name": [f"name{chr(97 + i)}" for i in range(rows)],
            }
        )
        dataset = self.make_dataset(df)
        step = ActDropIdLikeColumns()

        self.assertTrue(step.suitable(dataset))

        self.fit_step(step, dataset)

        self.assertCountEqual(step.columns_to_drop, ["user_id", "session"])
        self.assertEqual(len(step.columns_to_drop), len(step.explanations))
        self.assertTrue(any("user_id" in expl for expl in step.explanations))
        self.assertTrue(any("session" in expl for expl in step.explanations))

        result = step.transform(dataset.X.copy())

        expected = df.drop(columns=["user_id", "session"])
        self.assertFrameEqual(result, expected)

    def test_unique_ratio_below_threshold_noop(self) -> None:
        ids = list(range(1000, 1024)) + [1023]
        df = pd.DataFrame(
            {
                "user_id": ids,
                "group": ["alpha"] * len(ids),
            }
        )
        step = ActDropIdLikeColumns()

        self.assertFalse(step.suitable(self.make_dataset(df)))

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
