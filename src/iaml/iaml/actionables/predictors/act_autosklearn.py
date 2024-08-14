"""
[STEP] Learn : AutoSkLearn
"""
import autosklearn.classification # pylint: disable=import-error
from ...predictor import Predictor
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step


# @is_step('predictor', 'tabular')
@is_step('to_compare')
class ActAutoSKLearn(Predictor):
    """
    [STEP] Learn : AutoSkLearn
    """
    name="Learn : AutoSkLearn"
    
    properties = [
        {
            'year': 2015,
            'name': 'Efficient and Robust Automated Machine Learning',
            'authors': [
                'Matthias Feurer',
                'Aaron Klein',
                'Katharina Eggensperger',
                'Jost Tobias Springenberg',
                'Manuel Blum',
                'Frank Hutter'
            ],
            'doi': 'https://doi.org/10.1007/978-3-030-05318-5_6 (V2)',
            # Store pages for book ? (2962--2970), Not always present in NeurIPS doc'
            'publisher': 'Advances in Neural Information Processing Systems 28 (NeurIPS 2015)'
        },
        {
            'year': 2020,
            'name': 'Auto-Sklearn 2.0: Hands-free AutoML via Meta-Learning',
            'authors': [
                'Matthias Feurer',
                'Katharina Eggensperger',
                'Stefan Falkner',
                'Marius Lindauer',
                'Frank Hutter'
            ],
            'doi': 'https://doi.org/10.48550/arXiv.2007.04074',
            'publisher': 'Journal of Machine Learning Research 23(261), 2022'
        }
    ]
    def __init__(self):
        self.configuration:dict = {
            'running_time': {
                'description': 'In seconds. Auto-SkLearn will search the best \
                    models during this time',
                'default': 30
            }
        }
        self.model:autosklearn.classification.AutoSklearnClassifier = None

        # Initialize reference for this step
        self._build_references(ActAutoSKLearn.properties)
    
    def fit(self, dataset: Dataset): # pylint: disable=unused-argument
        """
        Fit AutoSkLearn on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=self.get_config('running_time'),
            max_models_on_disc=5,
            memory_limit = 102400)
        
        self.model.fit(dataset.X, dataset.y)
        
        return self


    def suitable(self, dataset:Dataset) -> bool:
        """
        Does this step suitable for this candidate

        Args:
            candidate (Candidate): Suitable for this candidate

        Returns:
            bool: Suitable ?
        """
        return dataset.type_of_target in \
            ['binary', 'multiclass',  'multilabel-indicator', 'continuous']
    
    def priorize(self, candidate:Candidate=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5 # neutral
