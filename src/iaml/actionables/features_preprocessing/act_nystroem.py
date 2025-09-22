"""[STEP] Decompose features with Nystroem"""
import textwrap
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActNystroem(Actionable):
    """[STEP] Apply Nystroem method for dimensionality reduction"""

    name: str = "Nystroem"
    _description: str = "Apply the Nystroem method for dimensionality reduction \
        over a list of columns"
    _description_long: str = textwrap.dedent('''\
        The Nystroem method is a technique used for approximating kernel methods, 
        which helps in reducing the computational cost of kernel-based algorithms.
        It approximates a kernel map using a subset of the data, making it suitable
        for large datasets. This approach enables dimensionality reduction by creating
        a low-rank approximation of the original kernel matrix.
    ''')

    def __init__(self):
        self.configuration = {
            'kernel': {
                'description': 'Kernel map to be approximated.',
                'default': 'rbf',
                'categorical': ["poly", "rbf", "sigmoid", "cosine", "chi2"]
            },
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [50, 10000]
            },
            'coef0': {
                'description': 'Zero coefficient for polynomial and sigmoid kernels.',
                'default': 0.0,
                'range': [-1.0, 1.0]
            },
            'degree': {
                'description': 'Degree of the polynomial kernel.',
                'default': 3,
                'range': [2, 5]
            },
            'gamma': {
                'description': textwrap.dedent('''\
                    Gamma parameter for the RBF, laplacian, polynomial,
                    exponential chi2 and sigmoid kernels.'''),
                'default': 0.1,
                'range': [3.06e-05, 8.0]
            },
            'random_state': {
                'description': 'Random State',
                'default': 42
            }
        }
        self.optimizable: bool = True
        self.preprocessor: bool = None

    def fit(self, dataset: Dataset) -> Actionable:
        self.preprocessor = Nystroem(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply Nystroem

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        return pd.DataFrame(self.preprocessor.transform(X))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5


    def suitable(self, dataset: Dataset) -> bool:
        if self.get_config('kernel') == 'chi2' and \
            not (dataset.X < 0).any().any():
            self.configure('kernel', 'rbf') # pylint: disable=too-many-function-args
        return True
