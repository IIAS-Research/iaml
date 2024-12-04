"""[STEP] Gaussian NB"""

import textwrap
from typing import Any
from sklearn.naive_bayes import GaussianNB
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step

@is_step('predictor', 'tabular', 'classifier')
class ActGaussianNb(Predictor):
    """[STEP] Gaussian NB"""

    name: str = " Gaussian NB"
    _description: str = textwrap.dedent('''\
        GaussianNB is a machine learning algorithm that makes predictions
        based on the Gaussian (normal) distribution of the input features.''')
    _description_long: str = textwrap.dedent('''\
        GaussianNB is a type of naive Bayes classifier that assumes the
        input features are independent and follow a Gaussian (normal) distribution.
        It uses Bayes' theorem to calculate the probability of each class given the
        input features and then makes a prediction based on the highest probability.
        GaussianNB is particularly useful when the input features have a continuous
        distribution and can be modeled well by a normal distribution.
        It is a simple and fast algorithm that works well for many classification problems,
        especially when the number of features is much larger than the number of samples.''')
    refs: list[dict[str, Any]] = []

    def __init__(self):
        self.configuration = {
            'var_smoothing': {
                'description': textwrap.dedent('''\
                    Portion of the largest variance of all features that is
                    added to variances for calculation stability.'''),
                'default': 1e-9,
                'range': [1e-11, 1e-4]
            },
        }
        self.model: GaussianNB = None

    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        self.model = GaussianNB(**self.passthrough_parameters())
        self.model.fit(dataset.X, dataset.y)
        return self

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5 # neutral
