"""Tests for ActMICEForestImputer."""
import unittest

import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

try:
    from iaml.actionables.cleaning.act_mice import ActMICEForestImputer
    MICE_AVAILABLE = True
except ImportError:
    ActMICEForestImputer = None
    MICE_AVAILABLE = False


@unittest.skipIf(not MICE_AVAILABLE, "miceforest dependency is not installed")
class TestActMICEForestImputer(StepTestCase):
    def test_imputes_numeric_columns_and_preserves_object(self) -> None:
        df = pd.DataFrame(
            {
                "age": [20.0, 21.0, np.nan, 23.0, 24.0, 25.0],
                "score": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0],
                "city": ["paris", "lyon", None, "dijon", "nice", "nancy"],
            }
        )
        step = ActMICEForestImputer()

        result = self.apply_transform(step, df)

        self.assertListEqual(result["city"].tolist(), df["city"].tolist())
        missing_age = int(df["age"].isna().sum())
        missing_score = int(df["score"].isna().sum())
        self.assertIsNotNone(step.kernel)
        self.assertEqual(result["age"].isna().sum(), 0)
        self.assertEqual(result["score"].isna().sum(), 0)
        for col in ("age", "score"):
            mask = df[col].notna()
            np.testing.assert_allclose(
                result.loc[mask, col].to_numpy(),
                df.loc[mask, col].to_numpy(),
            )
        self.assertTrue(any("`age`" in explanation for explanation in step.explanations))
        self.assertTrue(any("`score`" in explanation for explanation in step.explanations))

    def test_skips_when_dataset_too_small(self) -> None:
        df = pd.DataFrame(
            {
                "age": [1.0, np.nan, 3.0, 4.0],
                "score": [1.0, 2.0, np.nan, 4.0],
            }
        )
        step = ActMICEForestImputer()

        result = self.apply_transform(step, df)

        self.assertIsNone(step.kernel)
        self.assertTrue(any("trop petit" in explanation for explanation in step.explanations))
        self.assertFrameEqual(result, df, check_dtype=False)

    def test_prefills_all_nan_numeric_column(self) -> None:
        df = pd.DataFrame({"all_nan": [np.nan, np.nan, np.nan]})
        step = ActMICEForestImputer()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"all_nan": [0.0, 0.0, 0.0]})
        self.assertFrameEqual(result, expected)
