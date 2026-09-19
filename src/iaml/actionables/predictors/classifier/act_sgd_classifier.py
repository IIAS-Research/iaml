"""[STEP] SGD Classifier"""
import textwrap
from typing import Any
import numpy as np
from sklearn.linear_model import SGDClassifier
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActSGDClassifier(Predictor):
    """[STEP] SGD Classifier"""

    name: str = "SGD Classifier"
    _description: str = textwrap.dedent('''\
        SGDClassifier is a linear classifier optimized with stochastic gradient
        descent, suitable for large-scale classification tasks.''')
    _description_long: str = textwrap.dedent('''\
        SGDClassifier learns a linear decision boundary using stochastic gradient
        descent. It supports hinge loss (linear SVM) and logistic loss for
        probabilistic classification, making it efficient on large datasets.''')
    _usage: str = "Use when you need a fast linear classifier for large numeric tabular data, as a baseline vs ActDecisionTreeClassifier or ActExtraTreesClassifier. Applicable to binary, multiclass, or multilabel targets with numeric features. Avoid when data are mostly categorical or require complex nonlinear boundaries."
    refs: list[dict[str, Any]] = [
        {
            'year': 2010,
            'name': 'Large-Scale Machine Learning with Stochastic Gradient Descent',
            'authors': [
                'Leon Bottou'
            ],
            'doi': 'https://doi.org/10.1145/1796439.1796440',
            'publisher': 'Proceedings of COMPSTAT 2010'
        }
    ]

    def __init__(self):
        self.configuration = {
            'loss': {
                'description': 'The loss function to be used.',
                'default': 'log_loss',
                'categorical': ['log_loss', 'hinge']
            },
            'penalty': {
                'description': 'The penalty (regularization term) to be used.',
                'default': 'l2',
                'categorical': ['l2', 'l1', 'elasticnet']
            },
            'alpha': {
                'description': 'Constant that multiplies the regularization term.',
                'default': 0.0001,
                'range': [1e-07, 0.1]
            },
            'l1_ratio': {
                'description': 'The Elastic Net mixing parameter.',
                'default': 0.15,
                'range': [1e-09, 1.0]
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
            'learning_rate': {
                'description': 'Learning rate schedule.',
                'default': 'optimal',
                'categorical': ['optimal', 'invscaling', 'constant', 'adaptive']
            },
            'eta0': {
                'description': textwrap.dedent('''\
                    The initial learning rate for the "constant", "invscaling"
                    or "adaptive" schedules.'''),
                'default': 0.01,
                'range': [1e-07, 0.1]
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
                    When set to True, computes the averaged SGD weights across
                    all updates and stores the result in the coef_ attribute.'''),
                'default': False
            },
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            }
        }
        self.model: SGDClassifier = None
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
        self.model = SGDClassifier(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def predict_proba(self, X):
        if self.model is None:
            raise AttributeError("Model is not fitted")
        if getattr(self.model, 'loss', None) in ['log_loss', 'modified_huber']:
            return self.model.predict_proba(self._select_features(X))
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
