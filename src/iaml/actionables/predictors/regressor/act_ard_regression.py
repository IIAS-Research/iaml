"""
[STEP] ARD Regression
"""

import textwrap
from sklearn.linear_model import ARDRegression
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'regressor')
class ActARDRegression(Predictor):
    """
    [STEP] ARD Regression
    """
    name = "ARD Regression"
    _description = textwrap.dedent('''\
        ARDRegression is a powerful tool that helps computers make accurate
        predictions by giving each feature its own importance weight.''')
    _description_long = textwrap.dedent('''\
        ARDRegression (Automatic Relevance Determination Regression)
        is a Bayesian regression technique used for predicting continuous outcomes.
        It works by assigning weights to each feature, allowing some features to be
        more important than others.
        These weights are determined automatically during training,
        hence the "automatic relevance determination.''')
    _usage = "Use when you want Bayesian linear regression with automatic relevance on tabular data, as a sparse alternative to ActElasticNetRegressor. Applicable to continuous targets with many features. Avoid when strong nonlinearity or interactions suggest ActExtraTreesRegressor."
    refs = [
        {
            'year': 1996,
            'name': 'Bayesian Non-Linear Modeling for the Prediction Competition',
            'authors': ['David J. C. MacKay'],
            'doi': 'https://doi.org/10.1007/978-94-015-8729-7_18',
            'publisher': 'Springer, Dordrecht'
        }
    ]
    def __init__(self):
        self.configuration = {
            'alpha_1': {
                'description': textwrap.dedent('''\
                    Hyper-parameter : shape parameter for the Gamma
                    distribution prior over the alpha parameter.'''),
                'default': 1e-06,
                'range': [1e-10, 0.001]
            },
            'alpha_2': {
                'description': textwrap.dedent('''\
                    Hyper-parameter : inverse scale parameter (rate parameter)
                    for the Gamma distribution prior over the alpha parameter.'''),
                'default': 1e-06,
                'range': [1e-10, 0.001]
            },
            'lambda_1': {
                'description': textwrap.dedent('''\
                    Hyper-parameter : shape parameter for the Gamma
                    distribution prior over the lambda parameter.'''),
                'default': 1e-10,
                'range': [1e-10, 0.001]
            },
            'lambda_2': {
                'description': textwrap.dedent('''\
                    Hyper-parameter : inverse scale parameter (rate parameter)
                    for the Gamma distribution prior over the lambda parameter.'''),
                'default': 1e-10,
                'range': [1e-10, 0.001]
            },
            'threshold_lambda': {
                'description': textwrap.dedent('''\
                    Threshold for removing (pruning) weights with high
                    precision from the computation.'''),
                'default': 10000.0,
                'range': [1000.0, 100000.0]
            },
            'tol': {
                'description': 'Stop the algorithm if w has converged.',
                'default': 0.001,
                'range': [1e-05, 0.1]
            }
        }

        self.model: ARDRegression = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = ARDRegression(**self.passthrough_parameters())

        self.model.fit(dataset.X, dataset.y)

        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'continuous'

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
