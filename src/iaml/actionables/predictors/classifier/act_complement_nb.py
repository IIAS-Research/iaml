"""[STEP] Complement NB"""

import textwrap
from typing import Any
from sklearn.naive_bayes import ComplementNB
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActComplementNB(Predictor):
    """[STEP] Complement NB"""

    name: str = "Complement NB"
    _usage: str = "Use when you need a fast NB baseline for imbalanced count features vs ActBernoulliNb. Applicable to non-negative numeric data (counts/TF-IDF) for binary or multiclass classification. Avoid when features are negative/continuous or nonlinear patterns favor ActCatBoost."
    _description: str = textwrap.dedent('''\
        ComplementNB is a Naive Bayes variant tailored for imbalanced text
        classification by using statistics from the complement of each class.''')
    _description_long: str = textwrap.dedent('''\
        ComplementNB modifies the traditional multinomial Naive Bayes formula
        by estimating feature weights from the complement of each class. This
        reduces the bias toward frequent classes, making it effective for
        imbalanced text datasets and high-dimensional sparse features.''')
    refs: list[dict[str, Any]] = [
        {
            'year': 2003,
            'name': 'Tackling the Poor Assumptions of Naive Bayes Text Classifiers',
            'authors': [
                'Jason D. M. Rennie',
                'Lawrence Shih',
                'Jaime Teevan',
                'David R. Karger'
            ],
            'doi': 'https://dl.acm.org/doi/10.5555/944919.944939',
            'publisher': (
                'Proceedings of the 20th International Conference on Machine Learning '
                '(ICML), pages 616--623'
            )
        }
    ]

    def __init__(self):
        self.configuration = {
            'alpha': {
                'description': textwrap.dedent('''\
                    Additive (Laplace/Lidstone) smoothing parameter (set
                    alpha=0 for no smoothing).'''),
                'default': 1.0,
                'range': [0.01, 100.0]
            },
            'fit_prior': {
                'description': textwrap.dedent('''\
                    Whether to learn class prior probabilities or not. If
                    false, a uniform prior will be used.'''),
                'default': True,
                'categorical': [True, False]
            },
            'norm': {
                'description': textwrap.dedent('''\
                    Whether to normalize by the total feature counts per class
                    to reduce the influence of document length.'''),
                'default': False,
                'categorical': [True, False]
            }
        }
        self.model: ComplementNB = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = ComplementNB(**self.passthrough_parameters())
        self.model.fit(self._select_features(dataset.X), dataset.y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        if dataset.type_of_target not in ['binary', 'multiclass']:
            return False

        columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not columns:
            return False

        try:
            return not (dataset.X[columns] < 0).any().any()
        except (TypeError, ValueError):
            return False

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
