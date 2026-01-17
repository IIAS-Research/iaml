"""[STEP] Bagging Classifier"""
import textwrap
from typing import Any
from sklearn.ensemble import BaggingClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActBaggingClassifier(Predictor):
    """[STEP] Bagging Classifier"""

    name: str = "Bagging Classifier"
    _description: str = textwrap.dedent('''\
        BaggingClassifier is an ensemble method that combines multiple
        base learners trained on bootstrapped samples to reduce variance.''')
    _description_long: str = textwrap.dedent('''\
        BaggingClassifier (Bootstrap Aggregating) fits several base estimators on
        random subsets of the training data and optionally on random subsets of
        features. Predictions are aggregated by majority vote, yielding a more
        stable classifier that is less sensitive to noise.''')
    _usage: str = "Use when you need variance reduction on numeric features and want a simple ensemble, versus ActExtraTreesClassifier. Applicable to tabular binary/multiclass/multilabel numeric data. Avoid when you need a single interpretable tree or can use ActDecisionTreeClassifier."
    refs: list[dict[str, Any]] = [
        {
            'year': 1996,
            'name': 'Bagging Predictors',
            'authors': [
                'Leo Breiman'
            ],
            'doi': 'https://doi.org/10.1023/A:1018054314350',
            'publisher': 'Machine Learning Vol. 24 page 123--140'
        }
    ]

    def __init__(self):
        self.configuration = {
            'n_estimators': {
                'description': 'Number of base estimators in the ensemble.',
                'default': 50,
                'range': [5, 500]
            },
            'max_samples': {
                'description': 'Fraction of samples to draw for each base estimator.',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'max_features': {
                'description': 'Fraction of features to draw for each base estimator.',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'bootstrap': {
                'description': 'Whether samples are drawn with replacement.',
                'default': True,
                'categorical': [True, False]
            },
            'bootstrap_features': {
                'description': 'Whether features are drawn with replacement.',
                'default': False,
                'categorical': [True, False]
            },
            'oob_score': {
                'description': textwrap.dedent('''\
                    Whether to use out-of-bag samples to estimate generalization.'''),
                'default': False,
                'categorical': [True, False]
            },
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            }
        }
        self.model: BaggingClassifier = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset):
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            raise ValueError(
                "BaggingClassifier requires at least one numeric feature."
            )

        params = self.passthrough_parameters()
        if params.get('oob_score') and not params.get('bootstrap', True):
            raise ValueError("oob_score=True requires bootstrap=True.")
        self.model = BaggingClassifier(**params)
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def predict_proba(self, X):
        return super().predict_proba(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass', 'multilabel-indicator'] \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
