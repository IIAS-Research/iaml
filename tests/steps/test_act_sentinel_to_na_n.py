"""Tests for ActSentinelToNaN."""
import numpy as np
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionables.features_precleaning.act_sentinel_to_na_n import ActSentinelToNaN


class TestActSentinelToNaN(StepTestCase):
    def test_replaces_numeric_and_text_sentinels(self) -> None:
        df = pd.DataFrame(
            {
                "value": [1, -999, 3, -9999],
                "status": ["ok", "NA", " unknown ", " -999 "],
            }
        )
        step = ActSentinelToNaN()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame(
            {
                "value": [1.0, np.nan, 3.0, np.nan],
                "status": ["ok", np.nan, np.nan, np.nan],
            }
        )
        self.assertFrameEqual(result, expected)

    def test_empty_string_not_replaced_when_disabled(self) -> None:
        df = pd.DataFrame({"status": ["", "NA", "ok"]})
        step = ActSentinelToNaN()
        step.configure("include_empty_string", False)

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"status": ["", np.nan, "ok"]})
        self.assertFrameEqual(result, expected)

    def test_numeric_text_not_replaced_when_disabled(self) -> None:
        df = pd.DataFrame({"status": ["-999", "-999.0", "ok"]})
        step = ActSentinelToNaN()
        step.configure("numeric_in_text", False)

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"status": ["-999", "-999.0", "ok"]})
        self.assertFrameEqual(result, expected)
