"""Minimal candidates learn categorical encodings inside their pipeline."""
import pickle
import unittest

import numpy as np
import pandas as pd

from iaml import Candidate, Dataset
from iaml.flow import Const, PipelineSpec, compile_pipeline, use
from iaml.actionables.cleaning.act_ordinal_encoder import ActOrdinalEncoder
from iaml.actionables.predictors.classifier.act_catboost_classifier import ActCatBoost
from iaml.actionables.predictors.classifier.act_xgboost import ActXGBoost
from iaml.metrics import RocAucMetric
from iaml.splitters import kfold_splitter


class MinimalCategoricalEncodingTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            'age': np.linspace(20.0, 80.0, 40),
            'gender': ['M', 'F'] * 20,
            'admission_type': ['planned', 'emergency', 'transfer', 'planned'] * 10,
        })
        self.y = np.array([0, 1] * 20)

    def candidate(self, model):
        spec = PipelineSpec.default()
        spec.remove('main')
        spec.minimal_predictor.replace(model)
        dataset = Dataset(self.X, self.y)
        candidate = Candidate(dataset, metrics=[RocAucMetric()], main_metric=RocAucMetric())
        return compile_pipeline(spec).run(candidate)[0]

    def test_boosted_models_receive_numeric_features_with_or_without_missing_values(self):
        models = [use(ActXGBoost, n_estimators=Const(5), max_depth=Const(2)),
                  use(ActCatBoost, iterations=Const(5), depth=Const(2))]
        original = self.X.copy(deep=True)
        for model in models:
            for missing in (False, True):
                with self.subTest(model=model.component, missing=missing):
                    self.X = original.copy(deep=True)
                    if missing:
                        self.X.loc[0, 'age'] = np.nan
                        self.X.loc[1, 'gender'] = None
                    candidate = self.candidate(model)
                    encoders = [step for _, step in candidate.pipeline.training_steps
                                if isinstance(step, ActOrdinalEncoder)]
                    self.assertEqual(len(encoders), 1)
                    self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype)
                                        for dtype in candidate.dataset.X.dtypes))
                    scores = candidate.training_evaluate(
                        Dataset(self.X, self.y), splitter=kfold_splitter, cache_split=False)
                    self.assertTrue(np.isfinite(scores['ROC AUC']))

    def test_encoding_is_preserved_for_unknown_categories_and_serialization(self):
        candidate = self.candidate(use(ActXGBoost,
                                      n_estimators=Const(5), max_depth=Const(2)))
        candidate.pipeline.fit(self.X, self.y, metrics=[RocAucMetric()])
        probe = self.X.iloc[:3].copy()
        probe.loc[probe.index[0], 'gender'] = 'unknown'
        expected = candidate.pipeline.predict_proba(probe)
        restored = pickle.loads(pickle.dumps(candidate.pipeline))
        np.testing.assert_allclose(restored.predict_proba(probe), expected)
        self.assertTrue(np.isfinite(expected).all())


if __name__ == '__main__':
    unittest.main()
