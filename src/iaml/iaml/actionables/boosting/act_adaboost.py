"""
STEP
Apply AdaBoost on models
"""
from sklearn.ensemble import AdaBoostClassifier

from ...actionable import Actionable
from ...decorators.all import runner
from ...candidate import Candidate


class ActAdaBoost(Actionable):
    """
    Apply Adaboost on models
    
    Configuration:
        * `random_state`: Random seed (defaults to 42).
        * `n_estimator`: Number of estimators (defaults to 2000).
    """
    name = "AdaBoost Classifier"
    refs = [
        {
            'year': 1995,
            'name': (
                'A desicion-theoretic generalization of on-line learning'
                'and an application to boosting'
            ),
            'authors': [
                'Yoav Freund',
                'Robert E. Schapire'
            ],
            'doi': 'https://doi.org/10.1007/3-540-59119-2_166',
            'publisher': 'Springer, Berlin, Heidelberg'
        }
    ]

    def __init__(self):
        self.configuration = {
            'random_state': {
                'description': 'random_state',
                'default': 42
            },
            'n_estimator': {
                'description': 'Number of estimators',
                'default': 2000
            },
        }

    @runner
    def run(self, candidate: Candidate) -> Candidate:
        model = AdaBoostClassifier(
            candidate.model,
            n_estimators=self.get_config('n_estimator'),
            random_state=self.get_config('random_state'))

        model.fit(candidate.dataset.X, candidate.dataset.y)

        return candidate.to_output(None, None, model)

    def priorize(self, _: Candidate = None) -> float:
        return 0.5
