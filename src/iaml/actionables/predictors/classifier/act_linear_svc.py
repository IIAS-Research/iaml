"""[STEP] Linear SVC Classifier"""
import textwrap
from typing import Any
from sklearn.svm import LinearSVC
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActLinearSVC(Predictor):
    """[STEP] Linear SVC Classifier"""

    name: str = "Linear SVC Classifier"
    _description: str = textwrap.dedent('''\
        LinearSVC is a fast linear support vector classifier for high-dimensional
        feature spaces, fitting a linear decision boundary with a large margin.''')
    _description_long: str = textwrap.dedent('''\
        LinearSVC solves a linear SVM optimization problem with the liblinear
        optimizer. It scales well to sparse and high-dimensional data and supports
        multiclass classification via a one-vs-rest strategy.''')
    _usage: str = "Use when you need a fast linear margin classifier for high-dimensional numeric or sparse data; compare ActComplementNB. Applicable to tabular numeric binary or multiclass targets. Avoid when strong non-linear patterns or tree ensembles like ActExtraTreesClassifier are needed."
    refs: list[dict[str, Any]] = [
        {
            'year': 1995,
            'name': 'Support-Vector Networks',
            'authors': [
                'Corinna Cortes',
                'Vladimir Vapnik'
            ],
            'doi': 'https://doi.org/10.1007/BF00994018',
            'publisher': 'Machine Learning Vol. 20, No. 3, page 273--297'
        },
        {
            'year': 2008,
            'name': 'LIBLINEAR: A Library for Large Linear Classification',
            'authors': [
                'Rong-En Fan',
                'Kai-Wei Chang',
                'Cho-Jui Hsieh',
                'Xiang-Rui Wang',
                'Chih-Jen Lin'
            ],
            'doi': 'https://doi.org/10.1145/1390681.1390687',
            'publisher': 'Journal of Machine Learning Research Vol. 9, page 1871--1874'
        }
    ]

    def __init__(self):
        self.configuration = {
            'C': {
                'description': 'Inverse of regularization strength.',
                'default': 1.0,
                'range': [1e-03, 100.0]
            },
            'penalty': {
                'description': textwrap.dedent('''\
                    Specify the norm of the penalty. The "l1" penalty requires
                    loss="squared_hinge" and dual=False.'''),
                'default': 'l2',
                'categorical': ['l2', 'l1']
            },
            'loss': {
                'description': textwrap.dedent('''\
                    Loss function. The "hinge" loss requires penalty="l2" and dual=True.'''),
                'default': 'squared_hinge',
                'categorical': ['squared_hinge', 'hinge']
            },
            'dual': {
                'description': textwrap.dedent('''\
                    Select the dual optimization problem. Prefer True when
                    n_samples < n_features.'''),
                'default': True,
                'categorical': [True, False]
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
            },
            'max_iter': {
                'description': 'Maximum number of iterations.',
                'default': 1000,
                'range': [100, 5000]
            },
            'fit_intercept': {
                'description': 'Whether to fit the intercept term.',
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
            'random_state': {
                'description': 'Random state for reproducibility.',
                'default': 42
            }
        }
        self.model: LinearSVC = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = LinearSVC(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass', 'multilabel-indicator'] \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
