"""Tests for the Actionable base step."""
import pandas as pd

from .step_test_case import StepTestCase

from iaml.actionable import Actionable
from iaml.candidate import Candidate
from iaml.decorators.is_step import find_steps_by_tag
from iaml.step import Step


class NoOpTransform(Actionable):
    name = "NoOpTransform"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.copy()


class NoOpPredictor(Actionable):
    name = "NoOpPredictor"

    def fit(self, dataset):
        return self

    def predict(self, X: pd.DataFrame) -> list:
        return [0 for _ in range(len(X))]


class AddOneTransform(Actionable):
    name = "AddOneTransform"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["age"] = X["age"] + 1
        return X


class ShrinkingTransform(Actionable):
    name = "ShrinkingTransform"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.iloc[:1].copy()


class NoneTransform(Actionable):
    name = "NoneTransform"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return None


class DuplicateResampler(Actionable):
    name = "DuplicateResampler"

    def resample(self, X: pd.DataFrame, y: list) -> tuple[pd.DataFrame, list]:
        X_resampled = pd.concat([X, X], ignore_index=True)
        y_resampled = list(y) + list(y)
        return X_resampled, y_resampled


class IndexPreservingResampler(Actionable):
    name = "IndexPreservingResampler"

    def resample(self, X: pd.DataFrame, y: list) -> tuple[pd.DataFrame, list]:
        X_resampled = pd.concat([X, X], axis=0)
        y_resampled = list(y) + list(y)
        return X_resampled, y_resampled


class NoneResampler(Actionable):
    name = "NoneResampler"

    def resample(self, X: pd.DataFrame, y: list) -> tuple[pd.DataFrame, list]:
        return None


class ConstantPredictor(Actionable):
    name = "ConstantPredictor"

    def fit(self, dataset):
        self.fit_calls = getattr(self, "fit_calls", 0) + 1
        self.fit_dataset = dataset
        return self

    def predict(self, X: pd.DataFrame) -> list:
        self.predict_calls = getattr(self, "predict_calls", 0) + 1
        return [1 for _ in range(len(X))]


class CountingActionable(Actionable):
    def fit(self, dataset):
        self.fit_calls = getattr(self, "fit_calls", 0) + 1
        return self


class ConfigurableCountingActionable(Actionable):
    name = "ConfigurableCountingActionable"

    def __init__(self):
        super().__init__()
        self.configuration = {
            "offset": {
                "description": "offset used to vary cache key",
                "default": 0,
            },
        }
        self.default_configuration()

    def fit(self, dataset):
        self.fit_calls = getattr(self, "fit_calls", 0) + 1
        return self


class ToggleSuitableCountingActionable(CountingActionable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_suitable = True

    def suitable(self, dataset) -> bool:
        return self.is_suitable


class UnsuitableCountingActionable(CountingActionable):
    def suitable(self, dataset) -> bool:
        return False


class ColumnSuitableCountingActionable(CountingActionable):
    name = "ColumnSuitableCountingActionable"

    def suitable(self, dataset) -> bool:
        return "eligible" in dataset.X.columns


class UnsuitablePredictor(Actionable):
    name = "UnsuitablePredictor"

    def suitable(self, dataset) -> bool:
        return False

    def fit(self, dataset):
        self.fit_calls = getattr(self, "fit_calls", 0) + 1
        return self

    def predict(self, X: pd.DataFrame) -> list:
        self.predict_calls = getattr(self, "predict_calls", 0) + 1
        return [1 for _ in range(len(X))]


class MultiMethodActionable(Actionable):
    name = "MultiMethodActionable"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.transform_calls = getattr(self, "transform_calls", 0) + 1
        X = X.copy()
        X["age"] = X["age"] + 10
        return X

    def resample(self, X: pd.DataFrame, y: list) -> tuple[pd.DataFrame, list]:
        self.resample_calls = getattr(self, "resample_calls", 0) + 1
        X_resampled = pd.concat([X, X], ignore_index=True)
        y_resampled = list(y) + list(y)
        return X_resampled, y_resampled

    def predict(self, X: pd.DataFrame) -> list:
        self.predict_calls = getattr(self, "predict_calls", 0) + 1
        return [0 for _ in range(len(X))]


class CannotDisable(Actionable):
    can_be_disabled = False


class TestActionable(StepTestCase):
    def _make_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "age": pd.Series([1, 2], dtype="int64"),
                "city": pd.Series(["paris", "lyon"], dtype="object"),
            },
            index=pd.Index([10, 20], name="row_id"),
        )

    def _make_candidate(self) -> Candidate:
        df = self._make_dataframe()
        dataset = self.make_dataset(df, y=[0, 1])
        return Candidate(dataset)

    def _make_candidate_with_column(self, name: str, values: list[int]) -> Candidate:
        df = self._make_dataframe()
        df[name] = pd.Series(values, dtype="int64", index=df.index)
        dataset = self.make_dataset(df, y=[0, 1])
        return Candidate(dataset)

    def _make_candidate_with_pipeline(self) -> Candidate:
        candidate = self._make_candidate()
        candidate = candidate.add_to_pipeline(NoOpTransform())
        candidate = candidate.add_to_pipeline(NoOpPredictor())
        return candidate

    def test_is_registered_with_actionable_tag(self) -> None:
        self.assertIn(Actionable, Step.available_steps)
        self.assertIn("actionable", Step.available_steps[Actionable])
        self.assertIn(Actionable, find_steps_by_tag("actionable"))

    def test_run_preserves_input_pipeline_and_stack(self) -> None:
        candidate = self._make_candidate_with_pipeline()
        candidate.stacked_path = ["seed", "transform", "predictor"]
        step = Actionable()

        expected_steps = [name for name, _ in candidate.pipeline.steps]
        expected_X = candidate.dataset.X.copy()
        expected_y = candidate.dataset.y.copy()
        expected_stack = list(candidate.stacked_path)
        expected_columns_types = dict(candidate.dataset.columns_types)

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]
        self.assertIsNot(output, candidate)
        self.assertIsNot(output.dataset, candidate.dataset)
        self.assertIsNot(output.pipeline, candidate.pipeline)
        self.assertIsNot(output.stacked_path, candidate.stacked_path)
        self.assertListEqual(output.stacked_path, expected_stack)

        self.assertFrameEqual(output.dataset.X, expected_X)
        self.assertListEqual(output.dataset.y.tolist(), expected_y.tolist())
        self.assertEqual(output.dataset.columns_types, expected_columns_types)

        self.assertFrameEqual(candidate.dataset.X, expected_X)
        self.assertListEqual(candidate.dataset.y.tolist(), expected_y.tolist())
        self.assertEqual(candidate.dataset.columns_types, expected_columns_types)
        self.assertListEqual(candidate.stacked_path, expected_stack)

        self.assertEqual([name for name, _ in output.pipeline.steps], expected_steps)
        self.assertEqual([name for name, _ in candidate.pipeline.steps], expected_steps)

    def test_run_disabled_single_candidate_bypasses_execution(self) -> None:
        candidate = self._make_candidate()
        step = CountingActionable()
        step.enable = False

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], candidate)
        self.assertEqual(getattr(step, "fit_calls", 0), 0)

    def test_run_disabled_transform_preserves_pipeline(self) -> None:
        candidate = self._make_candidate_with_pipeline()
        step = AddOneTransform()
        step.enable = False

        expected_steps = [name for name, _ in candidate.pipeline.steps]
        expected_X = candidate.dataset.X.copy()
        expected_y = candidate.dataset.y.copy()
        expected_columns_types = dict(candidate.dataset.columns_types)

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], candidate)
        self.assertEqual([name for name, _ in candidate.pipeline.steps], expected_steps)
        self.assertFrameEqual(candidate.dataset.X, expected_X)
        self.assertListEqual(candidate.dataset.y.tolist(), expected_y.tolist())
        self.assertEqual(candidate.dataset.columns_types, expected_columns_types)

    def test_run_unsuitable_single_candidate_bypasses_execution(self) -> None:
        candidate = self._make_candidate()
        step = UnsuitableCountingActionable()

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], candidate)
        self.assertEqual(getattr(step, "fit_calls", 0), 0)

    def test_run_on_list_returns_distinct_outputs(self) -> None:
        candidates = [self._make_candidate_with_pipeline(), self._make_candidate_with_pipeline()]
        step = Actionable()

        result = step.run(candidates)

        self.assertEqual(len(result), 2)
        self.assertIsNot(result[0], candidates[0])
        self.assertIsNot(result[1], candidates[1])
        self.assertIsNot(result[0], result[1])
        self.assertIsNot(result[0].pipeline, result[1].pipeline)

        for output, source in zip(result, candidates):
            self.assertFrameEqual(output.dataset.X, source.dataset.X)
            self.assertListEqual(output.dataset.y.tolist(), source.dataset.y.tolist())
            self.assertIsNot(output.dataset, source.dataset)
            self.assertEqual(
                [name for name, _ in output.pipeline.steps],
                [name for name, _ in source.pipeline.steps],
            )

    def test_run_on_list_mixed_suitable_and_unsuitable_candidates(self) -> None:
        suitable = self._make_candidate_with_column("eligible", [1, 1])
        unsuitable = self._make_candidate()
        step = ColumnSuitableCountingActionable()

        result = step.run([suitable, unsuitable])

        self.assertEqual(len(result), 2)
        self.assertIsNot(result[0], suitable)
        self.assertIs(result[1], unsuitable)
        self.assertEqual(getattr(step, "fit_calls", 0), 1)
        self.assertFrameEqual(result[0].dataset.X, suitable.dataset.X)
        self.assertListEqual(result[0].dataset.y.tolist(), suitable.dataset.y.tolist())
        self.assertIsNot(result[0].dataset, suitable.dataset)

    def test_run_on_list_disabled_bypasses_all_candidates(self) -> None:
        candidates = [self._make_candidate(), self._make_candidate()]
        step = CountingActionable()
        step.enable = False

        result = step.run(candidates)

        self.assertEqual(len(result), 2)
        self.assertIs(result[0], candidates[0])
        self.assertIs(result[1], candidates[1])
        self.assertEqual(getattr(step, "fit_calls", 0), 0)

    def test_cache_keyed_by_candidate_identity(self) -> None:
        candidate_a = self._make_candidate()
        candidate_b = self._make_candidate()
        step = Actionable()

        first = step.run(candidate_a)[0]
        second = step.run(candidate_b)[0]

        self.assertIsNot(first, candidate_a)
        self.assertIsNot(second, candidate_b)
        self.assertIsNot(first, second)

        third = step.run(candidate_a)[0]
        fourth = step.run(candidate_b)[0]

        self.assertIs(third, first)
        self.assertIs(fourth, second)

    def test_cache_ignores_candidate_mutations(self) -> None:
        candidate = self._make_candidate()
        step = Actionable()
        original_X = candidate.dataset.X.copy()

        first = step.run(candidate)[0]

        candidate.dataset.X.loc[candidate.dataset.X.index[0], "age"] = 99

        second = step.run(candidate)[0]

        self.assertIs(second, first)
        self.assertFrameEqual(second.dataset.X, original_X)
        self.assertNotEqual(
            candidate.dataset.X.loc[candidate.dataset.X.index[0], "age"],
            original_X.loc[original_X.index[0], "age"],
        )

    def test_cache_respects_configuration_changes(self) -> None:
        candidate = self._make_candidate()
        step = ConfigurableCountingActionable()

        first = step.run(candidate)[0]

        self.assertEqual(getattr(step, "fit_calls", 0), 1)

        step.configure("offset", 1)
        second = step.run(candidate)[0]

        self.assertIsNot(second, first)
        self.assertEqual(getattr(step, "fit_calls", 0), 2)

        third = step.run(candidate)[0]

        self.assertIs(third, second)
        self.assertEqual(getattr(step, "fit_calls", 0), 2)

    def test_cache_disabled_skips_cached_results(self) -> None:
        candidate = self._make_candidate()
        step = CountingActionable(use_cache=False)

        first = step.run(candidate)[0]
        second = step.run(candidate)[0]

        self.assertIsNot(first, second)
        self.assertEqual(getattr(step, "fit_calls", 0), 2)
        self.assertEqual(len(step.caches), 0)

    def test_cache_bypassed_when_disabled(self) -> None:
        candidate = self._make_candidate()
        step = CountingActionable()

        first = step.run(candidate)[0]

        self.assertEqual(getattr(step, "fit_calls", 0), 1)
        self.assertEqual(len(step.caches), 1)

        step.enable = False
        second = step.run(candidate)[0]

        self.assertIs(second, candidate)
        self.assertIsNot(second, first)
        self.assertEqual(getattr(step, "fit_calls", 0), 1)
        self.assertEqual(len(step.caches), 1)

        step.enable = True
        third = step.run(candidate)[0]

        self.assertIs(third, first)
        self.assertEqual(getattr(step, "fit_calls", 0), 1)
        self.assertEqual(len(step.caches), 1)

    def test_cache_bypassed_when_unsuitable(self) -> None:
        candidate = self._make_candidate()
        step = ToggleSuitableCountingActionable()

        first = step.run(candidate)[0]

        self.assertEqual(getattr(step, "fit_calls", 0), 1)
        self.assertEqual(len(step.caches), 1)

        step.is_suitable = False
        second = step.run(candidate)[0]

        self.assertIs(second, candidate)
        self.assertIsNot(second, first)
        self.assertEqual(getattr(step, "fit_calls", 0), 1)
        self.assertEqual(len(step.caches), 1)

        step.is_suitable = True
        third = step.run(candidate)[0]

        self.assertIs(third, first)
        self.assertEqual(getattr(step, "fit_calls", 0), 1)

    def test_transform_applies_changes_to_output(self) -> None:
        candidate = self._make_candidate()
        step = AddOneTransform()
        original_X = candidate.dataset.X.copy()
        original_y = candidate.dataset.y.copy()
        expected_columns_types = dict(candidate.dataset.columns_types)
        expected_index = original_X.index

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]
        expected = original_X.copy()
        expected["age"] = expected["age"] + 1

        self.assertFrameEqual(output.dataset.X, expected)
        self.assertEqual(output.dataset.columns_types, expected_columns_types)
        self.assertTrue(output.dataset.X.index.equals(expected_index))
        self.assertEqual(output.dataset.X.index.name, expected_index.name)
        self.assertEqual(len(output.dataset.X), len(output.dataset.y))
        self.assertIsNot(output.dataset, candidate.dataset)
        self.assertIsNot(output.dataset.X, candidate.dataset.X)
        self.assertIsNot(output.pipeline, candidate.pipeline)
        self.assertEqual(len(output.pipeline.steps), 1)
        self.assertIsInstance(output.pipeline.steps[0][1], AddOneTransform)

        self.assertFrameEqual(candidate.dataset.X, expected)
        self.assertListEqual(candidate.dataset.y.tolist(), original_y.tolist())
        self.assertEqual(candidate.dataset.columns_types, expected_columns_types)
        self.assertEqual(len(candidate.dataset.X), len(candidate.dataset.y))
        self.assertEqual(len(candidate.pipeline.transformers), 1)
        self.assertIs(candidate.pipeline.transformers[0][1], step)

    def test_transform_appends_to_existing_pipeline(self) -> None:
        candidate = self._make_candidate_with_pipeline()
        step = AddOneTransform()
        existing_predictor = candidate.pipeline.predictor
        existing_transformer = candidate.pipeline.transformers[0]

        result = step.run(candidate)

        output = result[0]
        expected_steps = ["NoOpTransform", "AddOneTransform", "NoOpPredictor"]

        self.assertEqual([name for name, _ in candidate.pipeline.steps], expected_steps)
        self.assertEqual([name for name, _ in output.pipeline.steps], expected_steps)
        self.assertIs(candidate.pipeline.predictor, existing_predictor)
        self.assertIs(candidate.pipeline.transformers[0], existing_transformer)
        self.assertIs(candidate.pipeline.transformers[1][1], step)
        self.assertIsInstance(output.pipeline.transformers[1][1], AddOneTransform)
        self.assertIsInstance(output.pipeline.predictor[1], NoOpPredictor)

    def test_transform_invalid_output_raises(self) -> None:
        candidate = self._make_candidate()
        step = NoneTransform()

        with self.assertRaises((AttributeError, TypeError, ValueError)):
            step.run(candidate)

        self.assertEqual(len(candidate.pipeline.transformers), 0)
        self.assertEqual(len(candidate.pipeline.resamplers), 0)
        self.assertIsNone(candidate.pipeline.predictor)

    def test_transform_allows_row_count_change(self) -> None:
        candidate = self._make_candidate()
        step = ShrinkingTransform()
        original_X = candidate.dataset.X.copy()
        original_y = candidate.dataset.y.tolist()

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]

        self.assertEqual(len(output.dataset.X), 1)
        self.assertListEqual(output.dataset.y.tolist(), original_y)
        self.assertEqual(len(output.dataset.y), len(original_y))
        self.assertNotEqual(len(output.dataset.X), len(output.dataset.y))
        self.assertListEqual(output.dataset.X.index.tolist(), [original_X.index[0]])
        self.assertEqual(len(output.pipeline.steps), 1)
        self.assertIsInstance(output.pipeline.steps[0][1], ShrinkingTransform)

        self.assertEqual(len(candidate.dataset.X), 1)
        self.assertListEqual(candidate.dataset.y.tolist(), original_y)
        self.assertEqual(len(candidate.dataset.y), len(original_y))
        self.assertNotEqual(len(candidate.dataset.X), len(candidate.dataset.y))
        self.assertListEqual(candidate.dataset.X.index.tolist(), [original_X.index[0]])
        self.assertEqual(len(candidate.pipeline.transformers), 1)
        self.assertIs(candidate.pipeline.transformers[0][1], step)

    def test_resample_applies_changes_to_output(self) -> None:
        candidate = self._make_candidate()
        step = DuplicateResampler()
        original_X = candidate.dataset.X.copy()
        original_y = candidate.dataset.y.tolist()
        expected_columns_types = dict(candidate.dataset.columns_types)

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]
        expected_X = pd.concat([original_X, original_X], ignore_index=True)
        expected_y = original_y + original_y

        self.assertFrameEqual(output.dataset.X, expected_X)
        self.assertListEqual(output.dataset.y.tolist(), expected_y)
        self.assertEqual(output.dataset.columns_types, expected_columns_types)
        self.assertEqual(len(output.dataset.X), len(output.dataset.y))
        self.assertIsNot(output.pipeline, candidate.pipeline)
        self.assertEqual(len(output.pipeline.resamplers), 1)
        self.assertIsInstance(output.pipeline.resamplers[0][1], DuplicateResampler)

        self.assertFrameEqual(candidate.dataset.X, expected_X)
        self.assertListEqual(candidate.dataset.y.tolist(), expected_y)
        self.assertEqual(candidate.dataset.columns_types, expected_columns_types)
        self.assertEqual(len(candidate.dataset.X), len(candidate.dataset.y))
        self.assertEqual(len(candidate.pipeline.resamplers), 1)
        self.assertIs(candidate.pipeline.resamplers[0][1], step)

    def test_resample_preserves_index_and_column_types(self) -> None:
        candidate = self._make_candidate()
        step = IndexPreservingResampler()
        original_X = candidate.dataset.X.copy()
        expected_columns_types = dict(candidate.dataset.columns_types)

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]
        expected_X = pd.concat([original_X, original_X], axis=0)

        self.assertFrameEqual(output.dataset.X, expected_X)
        self.assertEqual(output.dataset.columns_types, expected_columns_types)
        self.assertEqual(output.dataset.X.index.name, original_X.index.name)
        self.assertEqual(len(output.dataset.X), len(output.dataset.y))
        self.assertEqual(len(output.pipeline.resamplers), 1)
        self.assertIsInstance(output.pipeline.resamplers[0][1], IndexPreservingResampler)

        self.assertFrameEqual(candidate.dataset.X, expected_X)
        self.assertEqual(candidate.dataset.columns_types, expected_columns_types)
        self.assertEqual(candidate.dataset.X.index.name, original_X.index.name)
        self.assertEqual(len(candidate.dataset.X), len(candidate.dataset.y))
        self.assertEqual(len(candidate.pipeline.resamplers), 1)
        self.assertIs(candidate.pipeline.resamplers[0][1], step)

    def test_resample_invalid_output_raises(self) -> None:
        candidate = self._make_candidate()
        step = NoneResampler()
        original_X = candidate.dataset.X.copy()
        original_y = candidate.dataset.y.tolist()
        original_columns_types = dict(candidate.dataset.columns_types)

        with self.assertRaises((TypeError, ValueError)):
            step.run(candidate)

        self.assertFrameEqual(candidate.dataset.X, original_X)
        self.assertListEqual(candidate.dataset.y.tolist(), original_y)
        self.assertEqual(candidate.dataset.columns_types, original_columns_types)
        self.assertEqual(len(candidate.pipeline.transformers), 0)
        self.assertEqual(len(candidate.pipeline.resamplers), 1)
        self.assertIs(candidate.pipeline.resamplers[0][1], step)
        self.assertIsNone(candidate.pipeline.predictor)
        self.assertEqual(getattr(step, "caches", []), [])

    def test_resample_appends_to_existing_pipeline(self) -> None:
        candidate = self._make_candidate_with_pipeline()
        candidate = candidate.add_to_pipeline(DuplicateResampler())
        step = IndexPreservingResampler()
        expected_steps = [name for name, _ in candidate.pipeline.steps]

        result = step.run(candidate)

        output = result[0]
        expected_resamplers = ["DuplicateResampler", "IndexPreservingResampler"]

        self.assertEqual([name for name, _ in candidate.pipeline.resamplers], expected_resamplers)
        self.assertEqual([name for name, _ in output.pipeline.resamplers], expected_resamplers)
        self.assertEqual([name for name, _ in candidate.pipeline.steps], expected_steps)
        self.assertEqual([name for name, _ in output.pipeline.steps], expected_steps)
        self.assertIs(candidate.pipeline.resamplers[1][1], step)
        self.assertIsInstance(output.pipeline.resamplers[1][1], IndexPreservingResampler)

    def test_predictor_adds_model(self) -> None:
        candidate = self._make_candidate()
        step = ConstantPredictor()
        original_X = candidate.dataset.X.copy()
        original_y = candidate.dataset.y.copy()
        expected_columns_types = dict(candidate.dataset.columns_types)

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]

        self.assertIsNot(output, candidate)
        self.assertIsNot(output.dataset, candidate.dataset)
        self.assertIsNot(output.pipeline, candidate.pipeline)
        self.assertIsNotNone(output.pipeline.predictor)
        self.assertIsInstance(output.pipeline.predictor[1], ConstantPredictor)
        self.assertEqual(len(output.pipeline.steps), 1)
        self.assertEqual(getattr(step, "fit_calls", 0), 1)
        self.assertIs(getattr(step, "fit_dataset", None), candidate.dataset)
        self.assertEqual(getattr(step, "predict_calls", 0), 0)

        self.assertIsNotNone(candidate.pipeline.predictor)
        self.assertIs(candidate.pipeline.predictor[1], step)
        self.assertFrameEqual(candidate.dataset.X, original_X)
        self.assertListEqual(candidate.dataset.y.tolist(), original_y.tolist())
        self.assertEqual(candidate.dataset.columns_types, expected_columns_types)

    def test_predictor_replaces_existing_predictor(self) -> None:
        candidate = self._make_candidate_with_pipeline()
        existing_predictor = candidate.pipeline.predictor
        step = ConstantPredictor()

        result = step.run(candidate)

        output = result[0]
        expected_steps = ["NoOpTransform", "ConstantPredictor"]

        self.assertEqual([name for name, _ in candidate.pipeline.steps], expected_steps)
        self.assertEqual([name for name, _ in output.pipeline.steps], expected_steps)
        self.assertIsNot(candidate.pipeline.predictor, existing_predictor)
        self.assertIs(candidate.pipeline.predictor[1], step)
        self.assertIsInstance(output.pipeline.predictor[1], ConstantPredictor)
        self.assertIsInstance(output.pipeline.transformers[0][1], NoOpTransform)

    def test_unsuitable_predictor_returns_input_candidate(self) -> None:
        candidate = self._make_candidate_with_pipeline()
        step = UnsuitablePredictor()
        expected_steps = [name for name, _ in candidate.pipeline.steps]
        existing_predictor = candidate.pipeline.predictor

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        self.assertIs(result[0], candidate)
        self.assertEqual([name for name, _ in candidate.pipeline.steps], expected_steps)
        self.assertIs(candidate.pipeline.predictor, existing_predictor)
        self.assertEqual(getattr(step, "fit_calls", 0), 0)
        self.assertEqual(getattr(step, "predict_calls", 0), 0)

    def test_multi_method_step_prefers_transform(self) -> None:
        candidate = self._make_candidate()
        step = MultiMethodActionable()
        original_X = candidate.dataset.X.copy()
        original_y = candidate.dataset.y.tolist()

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]
        expected = original_X.copy()
        expected["age"] = expected["age"] + 10

        self.assertFrameEqual(output.dataset.X, expected)
        self.assertListEqual(output.dataset.y.tolist(), original_y)
        self.assertEqual(len(output.pipeline.transformers), 1)
        self.assertEqual(len(output.pipeline.resamplers), 0)
        self.assertIsNone(output.pipeline.predictor)
        self.assertEqual(getattr(step, "transform_calls", 0), 1)
        self.assertEqual(getattr(step, "resample_calls", 0), 0)
        self.assertEqual(getattr(step, "predict_calls", 0), 0)

    def test_cannot_disable_enforces_enable(self) -> None:
        candidate = self._make_candidate()
        step = CannotDisable()
        step.enable = False

        self.assertTrue(step.enable)
        step.enable = False
        self.assertTrue(step.enable)

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        self.assertIsNot(result[0], candidate)
        self.assertTrue(step.enable)

    def test_run_handles_empty_dataset(self) -> None:
        df = pd.DataFrame({"age": pd.Series([], dtype="int64")})
        dataset = self.make_dataset(df, y=[])
        candidate = Candidate(dataset)
        step = NoOpTransform()

        result = step.run(candidate)

        self.assertEqual(len(result), 1)
        output = result[0]
        self.assertFrameEqual(output.dataset.X, df)
        self.assertListEqual(output.dataset.y.tolist(), [])
