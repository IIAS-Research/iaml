"""[STEP] Coxnet Survival Analysis"""
import inspect
import textwrap
from typing import Any
from sksurv.linear_model import CoxnetSurvivalAnalysis

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....data_type import DataType
from ....decorators.all import is_step


@is_step('predictor', 'tabular', 'survival')
class ActCoxnetSurvivalAnalysis(Predictor):
    """[STEP] Coxnet Survival Analysis"""

    name: str = "CoxnetSurvivalAnalysis"
    _description: str = textwrap.dedent('''\
        CoxnetSurvivalAnalysis fits a Cox proportional hazards model with
        elastic-net regularization, combining L1 and L2 penalties to handle
        high-dimensional survival data.''')
    _description_long: str = textwrap.dedent('''\
        CoxnetSurvivalAnalysis estimates a Cox proportional hazards model while
        applying elastic-net regularization along a path of penalty strengths.
        The L1 component encourages sparse feature selection, while the L2
        component stabilizes coefficients when predictors are correlated. This
        makes the model well suited for survival datasets with many variables
        and right-censored observations.''')
    _usage: str = "Use when you need regularized Cox with feature selection; compare ActCox or ActRandomSurvivalForest. Applicable to tabular survival data with many numeric, correlated predictors. Avoid when effects are highly nonlinear or inputs are mostly categorical."
    refs: list[dict[str, Any]] = [
        {
            'year': 2011,
            'name': "Regularization Paths for Cox's Proportional Hazards Model \
                via Coordinate Descent",
            'authors': [
                'Noah Simon',
                'Jerome Friedman',
                'Trevor Hastie',
                'Robert Tibshirani'
            ],
            'doi': 'https://doi.org/10.18637/jss.v039.i05',
            'publisher': 'Journal of Statistical Software, 39(5)'
        },
        {
            'year': 2005,
            'name': 'Regularization and Variable Selection via the Elastic Net',
            'authors': [
                'Hui Zou',
                'Trevor Hastie'
            ],
            'doi': 'https://doi.org/10.1111/j.1467-9868.2005.00503.x',
            'publisher': 'Journal of the Royal Statistical Society Series B'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'l1_ratio': {
                'description': 'Mixing parameter between L1 and L2 penalty.',
                'default': 0.5,
                'range': [0.0, 1.0]
            },
            'n_alphas': {
                'description': 'Number of alpha values along the regularization path.',
                'default': 100,
                'range': [10, 200]
            },
            'alpha_min_ratio': {
                'description': textwrap.dedent('''\
                    Smallest alpha as a fraction of alpha_max for the regularization
                    path.'''),
                'default': 0.01,
                'range': [1e-04, 1.0]
            },
            'max_iter': {
                'description': 'Maximum number of coordinate descent iterations.',
                'default': 1000,
                'range': [100, 100000]
            },
            'tol': {
                'description': 'Stopping criterion.',
                'default': 1e-07,
                'range': [1e-09, 1e-03]
            },
            'fit_baseline_model': {
                'description': textwrap.dedent('''\
                    Fit baseline hazard models to enable survival function
                    predictions.'''),
                'default': False,
                'categorical': [True, False]
            }
        }
        self.model: CoxnetSurvivalAnalysis = None
        self.columns: list[str] = []

    def _select_features(self, X):
        if self.columns and hasattr(X, 'columns'):
            return X[self.columns]
        return X

    def _model_parameters(self) -> dict[str, Any]:
        params = self.passthrough_parameters()
        sig_params = inspect.signature(CoxnetSurvivalAnalysis).parameters
        return {key: value for key, value in params.items() if key in sig_params}

    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        self.columns = dataset.get_columns_names_by_type(DataType.NUMERIC)
        if not self.columns:
            self.columns = dataset.features

        self.model = CoxnetSurvivalAnalysis(**self._model_parameters())
        X, y = dataset.to_survival()
        self.model.fit(self._select_features(X), y)
        return self

    def predict(self, X):
        return super().predict(self._select_features(X))

    def predict_survival_function(self, X):
        return super().predict_survival_function(self._select_features(X))

    def predict_cumulative_hazard_function(self, X):
        return super().predict_cumulative_hazard_function(self._select_features(X))

    def score(self, X, y=None, *args, **kwargs):
        return self.model.score(self._select_features(X), y, *args, **kwargs)

    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival' \
            and bool(dataset.get_columns_names_by_type(DataType.NUMERIC))

    def priorize(self, candidate: Candidate = None) -> float:
        return 0.5  # neutral
