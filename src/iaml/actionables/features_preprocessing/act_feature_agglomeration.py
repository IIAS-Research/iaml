"""[STEP] Decompose features with FeatureAgglomeration"""
import textwrap
import pandas as pd
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


@is_step('features_preprocessing')
class ActFeatureAgglomeration(Actionable):
    """[STEP] Apply FeatureAgglomeration for dimensionality reduction"""
    name: str = "FeatureAgglomeration"
    _description: str = "Process FeatureAgglomeration algorithm over a set of features"
    _description_long: str = textwrap.dedent('''\
        FeatureAgglomeration is a clustering-based dimensionality reduction technique.
        It groups similar features together using a hierarchical clustering approach,
        which can help reduce the dimensionality of the dataset while preserving
        essential information. This technique is unsupervised, meaning it does not
        require labeled data, as it identifies clusters of features based on similarity.
    ''')
    _usage: str = "Use when you want to cluster highly correlated numeric features for dimensionality reduction, instead of ActKernelPCA or ActFastICA. Applicable to wide tabular data with many continuous features and no labels. Avoid when features are mostly categorical or you need interpretable original features."

    def __init__(self):
        self.configuration: dict = {
            'n_clusters': {
                'description': 'The number of clusters to find',
                'default': 25,
                'range': [2, 400]
                },
            'metric': {
                'description': 'Metric used to compute the linkage.',
                'default': 'euclidean',
                'categorical': ['euclidean']
                # 'categorical': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
                },
            'linkage': {
                'description': 'Which linkage criterion to use.',
                'default': 'ward',
                'categorical': ['ward', 'complete', 'average', 'single']
                # 'categorical': ['ward', 'complete', 'average', 'single']
                },
            'pooling_func': {
                'description': 'Which linkage criterion to use.',
                'default': np.mean,
                'categorical': [np.mean, np.median, np.max]
                }
            }

        self.optimizable: bool = True
        self.preprocessor: bool = None

    def fit(self, dataset: Dataset) -> Actionable:
        # pylint: disable=too-many-function-args
        self.configure('n_clusters', min(self.get_config('n_clusters'), dataset.X.shape[1]))

        self.preprocessor = FeatureAgglomeration(**self.passthrough_parameters())
        self.preprocessor.fit(dataset.X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply FeatureAgglomeration

        :param pd.DataFrame X: DataFrame to transform
        :return: Transformed dataset
        """
        return pd.DataFrame(self.preprocessor.transform(X))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
