"""Tests for ActSelectKBest."""
import pandas as pd

from iaml.actionables.features_selection.act_select_k_best import ActSelectKBest

from .step_test_case import StepTestCase


class TestActSelectKBest(StepTestCase):
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

    def test_suitable_rejects_negative_values_for_chi2(self) -> None:
        df = pd.DataFrame({"a": [0, -1, 2], "b": [1, 2, 3]})
        dataset = self.make_dataset(df, y=[0, 1, 0])
        step = ActSelectKBest()
        step.configure("score_func", "chi2")

        self.assertFalse(step.suitable(dataset))
