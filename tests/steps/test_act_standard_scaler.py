"""Tests for ActStandardScaler."""
import pandas as pd

from iaml import ActStandardScaler, Candidate, Step
from iaml.decorators.is_step import find_steps_by_tag
from .step_test_case import StepTestCase


class TestActStandardScaler(StepTestCase):
    def test_normal_import_registers_a_serializable_step(self) -> None:
        step = ActStandardScaler()

        self.assertIn(ActStandardScaler, find_steps_by_tag("normalize"))
        self.assertEqual(step.tags, {"normalize"})
        restored = Step.from_pipeline(step.json_pipeline())
        self.assertIsInstance(restored, ActStandardScaler)
        self.assertTrue(restored.enable)

    def test_scales_numeric_columns_only(self) -> None:
        df = pd.DataFrame(
            {
                "age": [1, 2, 3],
                "score": [2.0, 4.0, 6.0],
                "city": ["paris", "lyon", "nice"],
            }
        )
        step = ActStandardScaler()

        result = self.apply_transform(step, df)

        numeric = df[["age", "score"]]
        scaled = (numeric - numeric.mean()) / numeric.std(ddof=0)
        expected = pd.DataFrame(
            {
                "age": scaled["age"],
                "score": scaled["score"],
                "city": df["city"],
            }
        )
        self.assertFrameEqual(result, expected, atol=1e-6, rtol=1e-6)

    def test_transform_uses_fitted_parameters(self) -> None:
        train = pd.DataFrame(
            {
                "age": [0, 10],
                "score": [1.0, 3.0],
                "city": ["a", "b"],
            }
        )
        step = ActStandardScaler()
        dataset = self.make_dataset(train)
        self.fit_step(step, dataset)

        new = pd.DataFrame(
            {
                "age": [5, 10],
                "score": [2.0, 3.0],
                "city": ["c", "d"],
            }
        )
        result = step.transform(new.copy())

        expected = pd.DataFrame(
            {
                "age": [0.0, 1.0],
                "score": [0.0, 1.0],
                "city": ["c", "d"],
            }
        )
        self.assertFrameEqual(result, expected, atol=1e-6, rtol=1e-6)

    def test_no_numeric_columns_returns_input(self) -> None:
        df = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        step = ActStandardScaler()

        result = self.apply_transform(step, df)

        expected = pd.DataFrame({"city": ["paris", "lyon"], "code": ["a", "b"]})
        self.assertFrameEqual(result, expected)

    def test_run_builds_a_pipeline_that_reuses_training_statistics(self) -> None:
        train = pd.DataFrame({"age": [0.0, 10.0], "city": ["a", "b"]})
        candidate = Candidate(self.make_dataset(train, [0, 1]))
        step = ActStandardScaler()

        outputs = step.run(candidate)

        self.assertEqual(len(outputs), 1)
        # The runner decorator wraps Step.run's Candidate return value in a list.
        result = outputs[0]  # pylint: disable=unsubscriptable-object
        self.assertFrameEqual(
            result.dataset.X,
            pd.DataFrame({"age": [-1.0, 1.0], "city": ["a", "b"]}),
        )
        self.assertEqual(len(result.pipeline.transformers), 1)
        self.assertIsInstance(result.pipeline.transformers[0][1], ActStandardScaler)

        new = pd.DataFrame({"age": [5.0, 15.0], "city": ["c", "d"]})
        self.assertFrameEqual(
            result.pipeline.transform(new),
            pd.DataFrame({"age": [0.0, 2.0], "city": ["c", "d"]}),
        )
