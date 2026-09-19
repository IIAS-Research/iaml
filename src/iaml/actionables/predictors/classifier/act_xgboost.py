"""[STEP] XGBoost classifier."""
import textwrap
from typing import Any

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from xgboost import XGBClassifier

from .._xgboost import xgboost_features
from ....predictor import Predictor
from ....dataset import Dataset
from ....candidate import Candidate
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'classifier', 'minimal_predictor')
class ActXGBoost(Predictor):
    """[STEP] XGBoost classifier."""

    name: str = "XGBoost"
    _description: str = textwrap.dedent('''\
        XGBoost trains an ensemble of boosted decision trees for binary
        or multiclass classification.''')
    _description_long: str = textwrap.dedent('''\
        XGBoost adds trees sequentially to improve predictions, using regularized
        gradient boosting and histogram-based split finding. Class labels are
        encoded during training and restored when predicting.''')
    _usage: str = "Use for boosted-tree classification on numeric or encoded tabular features. Supports binary and multiclass targets."
    refs: list[dict[str, Any]] = [
        {
            'name': 'XGBoost: A Scalable Tree Boosting System',
            'year': 2016,
            'authors': ['Tianqi Chen', 'Carlos Guestrin'],
            'doi': 'https://doi.org/10.1145/2939672.2939785',
            'publisher': 'Proceedings of the 22nd ACM SIGKDD International Conference, pages 785–794'
        }
    ]

    def __init__(self):
        self.configuration = {
            'max_depth': {
                'description': 'Maximum depth of each tree.',
                'default': 6,
                'range': [1, 16]
            },
            'random_state': {
                'description': 'Random seed for reproducibility.',
                'default': 42
            },
            'learning_rate': {
                'description': 'Shrinkage applied to each boosting round.',
                'default': 0.1,
                'range': [1e-3, 1.0]
            },
            'subsample': {
                'description': 'Fraction of training rows sampled for each tree.',
                'default': 1.0,
                'range': [0.1, 1.0]
            },
            'n_estimators': {
                'description': 'Number of boosting rounds.',
                'default': 100,
                'range': [1, 500]
            },
            'min_child_weight': {
                'description': 'Minimum sum of instance Hessians needed in a child.',
                'default': 1.0,
                'range': [0.0, 20.0]
            },
            'colsample_bytree': {
                'description': 'Fraction of features sampled for each tree.',
                'default': 1.0,
                'range': [0.1, 1.0]
            }
        }
        self.model: XGBClassifier = None
        self.label_encoder = LabelEncoder()

    def fit(self, dataset: Dataset):
        check_classification_targets(dataset.y)
        encoded_target = self.label_encoder.fit_transform(dataset.y)
        # IAML schedules candidates in parallel; keep each estimator single-threaded.
        self.model = XGBClassifier(
            n_jobs=1, tree_method='hist', **self.passthrough_parameters()
        )
        self.model.fit(xgboost_features(dataset.X), encoded_target)
        return self

    def predict(self, X):
        """Predict original class labels using XGBoost-safe feature names."""
        return super().predict(xgboost_features(X))

    def predict_proba(self, X):
        """Predict class probabilities using XGBoost-safe feature names."""
        return super().predict_proba(xgboost_features(X))

    @property
    def classes_(self):
        """Original labels, in the order of predict_proba columns."""
        return self.label_encoder.classes_

    def score(self, X, y, sample_weight=None):
        """Compute accuracy with the original class labels."""
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target in ['binary', 'multiclass']

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5
