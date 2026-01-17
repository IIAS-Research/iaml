"""[STEP] Passive Aggressive Classifier"""
import textwrap
from typing import Any
import numpy as np
from sklearn.linear_model import PassiveAggressiveClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActPassiveAggressiveClassifier(Predictor):
    """[STEP] Passive Aggressive Classifier"""

    name: str = "Passive Aggressive Classifier"
    _usage: str = "Use when you need a fast online linear classifier for large or streaming data. Applicable to numeric tabular binary/multiclass or multilabel targets. Avoid when patterns are nonlinear or categorical-heavy; prefer ActDecisionTreeClassifier or ActExtraTreesClassifier."
    _description: str = textwrap.dedent('''\
        PassiveAggressiveClassifier is an online linear classifier that updates
        only on misclassified samples, making it suitable for streaming or
        large-scale data.''')
    _description_long: str = textwrap.dedent('''\
        PassiveAggressiveClassifier is an online learning algorithm that performs
        aggressive updates when samples are misclassified and stays passive otherwise.
        It learns a linear decision boundary efficiently with hinge-style losses and
        supports multiclass problems with a one-vs-rest strategy.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2006,
            'name': 'Online Passive-Aggressive Algorithms',
            'authors': [
                'Koby Crammer',
                'Ofer Dekel',
                'Joseph Keshet',
                'Shai Shalev-Shwartz',
                'Yoram Singer'
            ],
            'doi': 'https://doi.org/10.5555/1248547.1248566',
            'publisher': 'Advances in Neural Information Processing Systems 19'
        }
    ]

    def __init__(self):
        self.configuration = {
            'C': {
                'description': 'Maximum step size (regularization strength).',
                'default': 1.0,
                'range': [1e-03, 100.0]
            },
            'loss': {
                'description': 'The loss function to be used.',
                'default': 'hinge',
                'categorical': ['hinge', 'squared_hinge']
            },
            'max_iter': {
                'description': 'Maximum number of passes over the training data.',
                'default': 1000,
                'range': [100, 5000]
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.001,
                'range': [1e-05, 0.1]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
                'default': True,
                'categorical': [True, False]
            },
            'shuffle': {
                'description': 'Whether to shuffle the training data after each epoch.',
                'default': True,
                'categorical': [True, False]
            },
            'class_weight': {
                'description': textwrap.dedent('''\
                    The "balanced" mode uses values of y to adjust weights inversely
                    proportional to class frequencies.'''),
                'default': None,
                'categorical': [None, 'balanced']
            },
            'average': {
                'description': textwrap.dedent('''\
                    When set to True, computes the averaged weights across all updates.'''),
                'default': False,
                'categorical': [True, False]
            },
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            }
        }
        self.model: PassiveAggressiveClassifier = None
        self.columns: list[str] = []
        self.target_type: str | None = None

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def _decision_function_proba(self, scores: np.ndarray) -> np.ndarray:
        if scores.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            return np.column_stack([1.0 - probs_pos, probs_pos])
        if self.target_type == 'multilabel-indicator':
            return 1.0 / (1.0 + np.exp(-scores))
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.target_type = dataset.type_of_target
        self.model = PassiveAggressiveClassifier(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def predict_proba(self, X):
        if self.model is None:
            raise AttributeError("Model is not fitted")
        scores = self.model.decision_function(self._select_features(X))
        return self._decision_function_proba(np.asarray(scores))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass', 'multilabel-indicator'] \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
