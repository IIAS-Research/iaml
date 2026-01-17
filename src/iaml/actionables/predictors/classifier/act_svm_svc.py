"""[STEP]  SVM Classifier"""
import textwrap
from typing import Any
from sklearn import svm
from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier')
class ActSVMSVC(Predictor):
    """[STEP]  SVM Classifier"""

    name: str = "SVM Classification"
    _description: str = textwrap.dedent('''\
        SVM Classifier is a machine learning algorithm that models the relationship
        between input features and a categorical output variable using a support vector machine
        (SVM). It can handle non-linearly separable data by using a kernel function to map the data
        into a higher-dimensional space.''')
    _description_long: str = textwrap.dedent('''\
        SVM Classifier is a type of classification algorithm that models the
        relationship between input features and a categorical output variable using a support vecto
        machine (SVM). It works by finding the optimal hyperplane or boundary that separates the
        data into different classes with the maximum margin.''')
    _usage: str = "Use when tabular classes need nonlinear boundaries on small-to-medium data; consider ActCatBoost or ActExtraTreesClassifier for baseline alternatives. Applicable to binary, multiclass, or multilabel targets. Avoid when data is huge, very sparse, or interpretability is required."
    refs: list[dict[str, Any]] = [
        {
            'year': 1999,
            'name': 'Probabilistic Outputs for Support Vector Machines and Comparisons to \
                Regularized Likelihood Methods',
            'authors': [
                'John C. Platt'
            ],
            'doi': "https://api.semanticscholar.org/CorpusID:56563878",
            'publisher': 'Microsoft Research'
        },
        {
            'year': 2001,
            'name': 'LIBSVM: A Library for Support Vector Machines',
            'authors': [
                'Chih-Chung Chang',
                'Chih-Jen Lin'
            ],
            'doi': 'https://doi.org/10.1145/1961189.1961199',
            'publisher': 'ACM Transactions on Intelligen Systems and Technology Vol.2 page 1--27'
        },
    ]

    def __init__(self):
        self.configuration = {
            'kernel': {
                'description': 'Kernel to use in the SVM',
                'default': 'rbf',
                'categorical': ['linear', 'poly', 'rbf', 'sigmoid']
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'class_weight': {
                'description': 'Can be set on "balanced" to improve results on unbalanced data',
                'default': None,
                'categorical': [None, 'balanced']
            },
            'tol': {
                'description': 'The stopping criterion.',
                'default': 0.001,
                'range': [1e-05, 0.1]
            }
        }
        self.model: svm.SVC = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = svm.SVC(
            probability = True, # Needed to predict_proba (thus MetaLearner)
            **self.passthrough_parameters()
            )
        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
