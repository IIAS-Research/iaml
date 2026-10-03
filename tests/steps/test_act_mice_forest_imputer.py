"""Tests for ActMICEForestImputer."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


@unittest.skipIf(not MICE_AVAILABLE, "miceforest dependency is not installed")
class TestMICEConfiguration(StepTestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame({
            "age": [20.0, np.nan, 22.0, 23.0, 24.0, 25.0],
            "binary": ["a", "b", None, "a", "b", "a"],
            "three_categories": ["a", "b", "c", None, "a", "b"],
        })

    def make_kernel(self) -> Mock:
        kernel = Mock()
        kernel.models = {"model": SimpleNamespace(params={})}
        kernel.impute_new_data.side_effect = lambda **kwargs: SimpleNamespace(
            complete_data=lambda dataset: kwargs["new_data"].ffill().bfill()
        )
        return kernel

    def test_configured_parameters_reach_fit_and_transform(self) -> None:
        for seed in (17, None):
            with self.subTest(random_state=seed):
                step = ActMICEForestImputer()
                step.configure({"max_iter": 2, "random_state": seed,
                                "auto_categorize": True,
                                "auto_categorize_max_cardinality": 2})
                kernel = self.make_kernel()
                with patch("iaml.actionables.cleaning.act_mice.mf.ImputationKernel",
                           return_value=kernel) as factory:
                    self.assertTrue(step.suitable(self.make_dataset(self.df)))
                    step.fit(self.make_dataset(self.df))
                    kernel.models["model"].params.pop("seed")
                    result = step.transform(self.df)

                fitted = factory.call_args.kwargs["data"]
                self.assertEqual(list(fitted.columns), ["age", "binary"])
                self.assertIsInstance(fitted["binary"].dtype, pd.CategoricalDtype)
                self.assertEqual(factory.call_args.kwargs["random_state"], seed)
                kernel.mice.assert_called_once_with(
                    2, n_jobs=step._n_jobs, verbose=False, seed=seed, random_state=seed
                )
                transformed = kernel.impute_new_data.call_args.kwargs
                self.assertEqual(transformed["iterations"], 2)
                self.assertIsInstance(transformed["new_data"]["binary"].dtype,
                                      pd.CategoricalDtype)
                self.assertEqual(kernel.models["model"].params["seed"],
                                 seed if seed is not None else 0)
                self.assertFalse(result[["age", "binary"]].isna().any().any())
                pd.testing.assert_series_equal(result["three_categories"],
                                               self.df["three_categories"])

    def test_unconfigured_parameters_keep_defaults(self) -> None:
        step = ActMICEForestImputer()
        kernel = self.make_kernel()
        with patch("iaml.actionables.cleaning.act_mice.mf.ImputationKernel",
                   return_value=kernel) as factory:
            result = self.apply_transform(step, self.df)

        self.assertEqual(list(factory.call_args.kwargs["data"].columns), ["age"])
        self.assertEqual(factory.call_args.kwargs["random_state"], 0)
        kernel.mice.assert_called_once_with(
            5, n_jobs=step._n_jobs, verbose=False, seed=0, random_state=0
        )
        self.assertEqual(kernel.impute_new_data.call_args.kwargs["iterations"], 5)
        pd.testing.assert_series_equal(result["binary"], self.df["binary"])

    def test_retries_preserve_configured_iterations_and_seed(self) -> None:
        step = ActMICEForestImputer()
        step.configure({"max_iter": 2, "random_state": 19})
        first = self.make_kernel()
        first.mice.side_effect = IndexError("mean matching failed")
        fallback = self.make_kernel()
        completed = SimpleNamespace(complete_data=lambda dataset: self.df[["age"]].ffill())
        fallback.impute_new_data.side_effect = [KeyError("seed"), completed]
        with patch("iaml.actionables.cleaning.act_mice.mf.ImputationKernel",
                   side_effect=[first, fallback]) as factory:
            result = self.apply_transform(step, self.df)

        for kernel in (first, fallback):
            kernel.mice.assert_called_once_with(
                2, n_jobs=step._n_jobs, verbose=False, seed=19, random_state=19
            )
        for call in factory.call_args_list:
            self.assertEqual(call.kwargs["random_state"], 19)
        self.assertEqual(factory.call_args.kwargs["mean_match_candidates"], 0)
        for call in fallback.impute_new_data.call_args_list:
            self.assertEqual(call.kwargs["iterations"], 2)
        self.assertEqual(fallback.impute_new_data.call_args.kwargs["mean_match_candidates"], 0)
        self.assertFalse(result["age"].isna().any())
