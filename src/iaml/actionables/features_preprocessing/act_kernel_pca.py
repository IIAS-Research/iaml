"""[STEP] Decompose features with KernelPCA"""
import textwrap
import pandas as pd
from sklearn.decomposition import KernelPCA
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


def _is_numeric_matrix(values: pd.DataFrame) -> bool:
    if values.empty:
        return False
    for column in values.columns:
        if not pd.api.types.is_numeric_dtype(values[column]):
            return False
    return not values.isna().any().any()


@is_step('features_preprocessing')
class ActKernelPCA(Actionable):
    """[STEP] Apply KernelPCA for dimensionality reduction"""
    name = "KernelPCA"
    _description = "Perform Kernel Principal Component Analysis (KernelPCA) on a dataset"
    _description_long = textwrap.dedent('''\
        KernelPCA is a dimensionality reduction technique that extends Principal Component Analysis (PCA)
        using kernel methods. It projects data into a higher-dimensional space before performing PCA,
        enabling it to capture complex, non-linear structures in the data. KernelPCA is useful for reducing
        dimensionality while preserving intricate patterns and relationships within the data.
    ''')
    _usage = "Use when non-linear structure matters and linear reductions are insufficient; consider ActFastICA if you want independent components. Applicable to dense numeric, scaled features. Avoid when data is very large, sparse, or interpretability is required."

    refs = [
        {
            'year': 1997,
            'name': 'Kernel principal component analysis',
            'authors': [
                'Bernhard Schölkopf',
                'Alexander Smola',
                'Klaus-Robert Müller'    
            ],
            'doi': 'https://doi.org/10.1007/BFb0020217',
            'publisher': 'Springer, Berlin, Heidelberg'
        },
        {
            'year': 2003,
            'name': 'Learning to find pre-images',
            'authors': [
                'Jason Weston',
                'Bernhard Schölkopf',
                'Gökhan Bakir'
            ],
            'doi': 'https://proceedings.neurips.cc/paper_files/paper/2003/file/ \
                    ac1ad983e08ad3304a97e147f522747e-Paper.pdf',
            'publisher': 'Advances in neural information processing systems 16 (2004) page 449--456'
        },
        {
            'year': 2009,
            'name': 'Finding structure with randomness: Probabilistic algorithms for constructing \
                approximate matrix decompositions',
            'authors': [
                'Nathan Halko',
                'Per-Gunnar Martinsson',
                'Joel A. Tropp'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.0909.4061',
            'publisher': 'SIAM Rev., Survey and Review section, Vol.53, No.2 page 217--288'
        },
        {
            'year': 2011,
            'name': 'A randomized algorithm for the decomposition of matrices',
            'authors': [
                'Per-Gunnar Martinsson',
                'Vladimir Rokhlin',
                'Mark Tygert'
            ],
            'doi': 'https://doi.org/10.1016/j.acha.2010.02.003',
            'publisher': 'Applied and Computational Harmonic Analysis, Vol.30, No.1 page 47--68'
        }
    ]
    def __init__(self):
        self.configuration = {
            'kernel': {
                'description': 'Kernel used for PCA.',
                'default': 'rbf',
                'categorical': ['poly', 'rbf', 'sigmoid', 'cosine']
            },
            'n_components': {
                'description': 'Number of components to keep.',
                'default': 100,
                'range': [10, 2000]
            },
            'coef0': {
                'description': textwrap.dedent('''\
                    Independent term in poly and sigmoid kernels. Ignored by
                    other kernels.'''),
                'default': 1.0,
                'range': [-1.0, 1.0]
            },
            'degree': {
                'description': 'Degree for poly kernels. Ignored by other kernels.',
                'default': 3,
                'range': [2, 5]
            },
            'random_state': {
                'description': 'Random State',
                'default': 42
            }
        }
        self.optimizable: bool = True
        self.preprocessor: bool = None


    def fit(self, dataset: Dataset) -> Actionable:
        self.preprocessor = None
        if not _is_numeric_matrix(dataset.X):
            return self
        try:
            self.preprocessor = KernelPCA(**self.passthrough_parameters())
            self.preprocessor.fit(dataset.X)
        except ValueError:
            higher_gamma = 1 / dataset.X.shape[1] + 0.05
            self.preprocessor = KernelPCA(gamma=higher_gamma, **self.passthrough_parameters())
            self.preprocessor.fit(dataset.X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply KernelPCA

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """

        if self.preprocessor is None:
            return X
        return pd.DataFrame(self.preprocessor.transform(X))

    def suitable(self, dataset: Dataset) -> bool:
        return _is_numeric_matrix(dataset.X)

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
