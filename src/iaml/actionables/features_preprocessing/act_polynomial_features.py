"""[STEP] Decompose features with PolynomialFeatures"""
import textwrap
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('features_preprocessing')
class ActPolynomialFeatures(Actionable):
    """[STEP] Preprocess with PolynomialFeatures"""

    name: str = "Preprocess with PolynomialFeatures"
    _usage: str = "Use when you want explicit polynomial interactions for linear models; consider ActKernelPCA for projection-based nonlinearity. Applicable to numeric tabular features with moderate dimensionality. Avoid when feature count will explode or when ActKBinsDiscretizer is a better match."
    _description: str = textwrap.dedent('''\
        PolynomialFeatures creates new features by combining existing
        features mathematically. It squares, cubes, and multiplies features to
        create more complex patterns.''')
    _description_long: str = textwrap.dedent('''\
        PolynomialFeatures is a preprocessing technique that
        generates new features based on polynomial relationships between existing
        features. This helps capture non-linear relationships in the data that may
        not be apparent from the original features alone. PolynomialFeatures is
        particularly useful when you suspect the underlying relationship in your data
        might not be straightforward or linear.''')

    def __init__(self):
        self.configuration = {
            'include_bias': {
                'description': 'If True (default), then include a bias \
                    column, the feature in which all polynomial powers are zero',
                'default': True
                },
            'interaction_only': {
                'description': 'If True, only interaction features are produced',
                'default': False
                },
            'degree': {
                'description': 'Degree of the polynomial kernel.',
                'default': 3,
                'range': [2, 5]
                }
            }

        self.optimizable: bool = True
        self.preprocessor: bool = None

    def fit(self, dataset: Dataset) -> Actionable:

        self.preprocessor = PolynomialFeatures(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply PolynomialFeatures

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        return pd.DataFrame(self.preprocessor.transform(X))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
