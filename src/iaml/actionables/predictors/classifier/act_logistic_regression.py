"""[STEP]  Logistic Regression Classifier"""
import textwrap
from typing import Any
from sklearn.linear_model import LogisticRegression
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'fast_predictor', 'classifier', 'baseline_predictor')
class ActLogisticRegression(Predictor):
    """[STEP]  Logistic Regression Classifier"""

    name: str = "Logistic Regression Classifier"
    _description: str = textwrap.dedent('''\
        LogisticRegression is a machine learning algorithm
        that models the relationship between input features and a binary
        output variable using a logistic function.''')
    _description_long: str = textwrap.dedent('''\
        LogisticRegression is a type of classification algorithm that models
        the relationship between input features and a binary
        output variable using a logistic function.
        It works by finding the best-fitting line or hyperplane that
        maximizes the likelihood of the observed output variables given the input features.''')
    _usage: str = "Use when you need a fast linear baseline over ActDecisionTreeClassifier. Applicable to tabular binary or multiclass data with numeric or one-hot inputs. Avoid when nonlinear interactions dominate or max accuracy is required; prefer ActCatBoost."
    refs: list[dict[str, Any]] = [
        {
            'year': 1944,
            'name': 'Application of the Logistic Function to Bio-Essay',
            'authors': [
                'Joseph Berkson'
            ],
            'doi': 'https://doi.org/10.2307/2280041',
            'publisher': (
                'Journal of the American Statistical Association '
                'Vol. 39, No. 227, page 357--365'
            )
        },
        {
            'year': 1951,
            'name': 'Why I Prefer Logits to Probits',
            'authors': [
                'Joseph Berkson'
            ],
            'doi': 'https://doi.org/10.2307/3001655',
            'publisher': (
                'Biometrics '
                'Vol. 7, No. 4, page 327--339'
            )
        }
    ]

    def __init__(self):
        self.configuration = {
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'penalty': {
                'description': 'Specify the norm of the penalty',
                'default': 'l2',
                'categorical': ['l2', None]
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.0001,
                'range': [1e-05, 0.1]
            },
            'class_weight': {
                'description': textwrap.dedent('''\
                    The “balanced” mode uses the values of y to automatically
                    adjust weights inversely proportional to class frequencies.'''),
                'default': None,
                'categorical': [None, 'balanced']
            }
        }
        self.model: LogisticRegression = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = LogisticRegression(
            **self.passthrough_parameters()
            )
        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
