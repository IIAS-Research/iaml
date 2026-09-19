"""Tests for ActKMeansFeatures step."""
import sys
from pathlib import Path
import types
import unittest

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
IAML_PATH = SRC_PATH / "iaml"

if str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))


def _stub_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


_stub_package("iaml", IAML_PATH)
_stub_package("iaml.actionables", IAML_PATH / "actionables")
_stub_package(
    "iaml.actionables.features_preprocessing",
    IAML_PATH / "actionables" / "features_preprocessing",
)

from .step_test_case import StepTestCase

try:
    from iaml.actionables.features_preprocessing.act_k_means_features import (
        ActKMeansFeatures,
    )
except ModuleNotFoundError as exc:
    if exc.name == "sklearn":
        ActKMeansFeatures = None
    else:
        raise


class TestActKMeansFeatures(StepTestCase):
    @unittest.skipIf(ActKMeansFeatures is None, "scikit-learn is required")
    def test_transform_adds_distance_and_label_features(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [0.0, 0.2, 5.0, 5.2],
                "f2": [0.0, 0.1, 5.0, 5.1],
            }
        )
        step = ActKMeansFeatures()
        step.configure("n_clusters", 2)

        result = self.apply_transform(step, df)

        expected_columns = [
            "f1",
            "f2",
            "kmeans_cluster_0_dist",
            "kmeans_cluster_1_dist",
            "kmeans_cluster",
        ]
        self.assertEqual(list(result.columns), expected_columns)
        self.assertEqual(result.shape, (len(df), 5))
        distances = result[["kmeans_cluster_0_dist", "kmeans_cluster_1_dist"]]
        self.assertTrue(np.isfinite(distances.to_numpy()).all())
        self.assertTrue((distances >= 0).all().all())
        self.assertTrue(set(result["kmeans_cluster"].unique()).issubset({0, 1}))

    @unittest.skipIf(ActKMeansFeatures is None, "scikit-learn is required")
    def test_distance_mode_closest_adds_single_feature(self) -> None:
        df = pd.DataFrame(
            {
                "f1": [1.0, 1.5, 2.0, 8.0, 9.0],
                "f2": [1.0, 1.2, 1.8, 8.0, 9.0],
            }
        )
        step = ActKMeansFeatures()
        step.configure(
            {
                "n_clusters": 3,
                "distance_mode": "closest",
                "add_cluster_label": False,
            }
        )

        result = self.apply_transform(step, df)

        self.assertEqual(list(result.columns), ["f1", "f2", "kmeans_cluster_dist"])
        self.assertEqual(result.shape, (len(df), 3))
        self.assertTrue(np.isfinite(result["kmeans_cluster_dist"].to_numpy()).all())

    @unittest.skipIf(ActKMeansFeatures is None, "scikit-learn is required")
    def test_transform_noop_when_no_active_columns(self) -> None:
        df = pd.DataFrame({"f1": [1, 1, 1], "f2": [2, 2, 2]})
        step = ActKMeansFeatures()

        result = self.apply_transform(step, df)

        self.assertFrameEqual(result, df)
