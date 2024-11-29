"""
[STEP] Aalen's Additive Model for Survival Analysis
"""
import textwrap
from lifelines import AalenAdditiveFitter
import pandas as pd

from ....predictor import Predictor
from ....candidate import Candidate
from ....dataset import Dataset
from ....decorators.all import is_step

# @is_step('predictor', 'tabular', 'survival')
@is_step('disabled')
class ActAalenAdditiveFitter(Predictor):
    """
    [STEP] Aalen's Additive Model for Survival Analysis
    """
    name = "AalenAdditiveFitter"
    description = textwrap.dedent('''\
        Aalen's Additive Model is a semi-parametric survival analysis model 
        that estimates survival time as a function of covariates, using a linear combination 
        of time-varying covariate effects. The additive nature of the model allows it to 
        account for time-varying effects of covariates on the hazard function.''')
    description_long = textwrap.dedent('''\
        Aalen's Additive Model is a flexible alternative to the Cox 
        Proportional Hazards model, providing time-varying covariate effects. The model 
        uses an additive approach to model the hazard function, making fewer assumptions 
        than proportional hazards models. It is particularly useful in situations where 
        covariate effects are expected to vary over time, and efficiently handles censored 
        data. The model estimates a baseline hazard function and additive contributions 
        of covariates, allowing for a more dynamic understanding of survival probabilities 
        over time.''')
    
    refs = [
        {
            'year': 2001,
            'name': 'Aalen’s Additive Model',
            'authors': [
                'O. Borgan',
                'J. Aalen',
                'H. Fekjær'
            ],
            'doi': 'https://doi.org/10.1007/978-1-4757-3462-1_4',
            'publisher': 'Survival and Event History Analysis, pages 109-142'
        }
    ]

    def __init__(self):
        self.configuration: dict = {
            'penalizer': {
                'description': 'The penalizer controls the amount of L2 regularization.',
                'default': 0.0,
                'range': [0.0, 1.0],
                'passthrough': False
            }
        }
        self.model: AalenAdditiveFitter = None
        
    def fit(self, dataset: Dataset):  # pylint: disable=unused-argument
        """
        Fit Aalen's Additive Model on Candidate.dataset

        Args:
            dataset (Dataset): Fit data

        Returns:
            Candidate: Transformed candidate
        """
        self.model = AalenAdditiveFitter(
            **self.passthrough_parameters()
        )
        
        y = pd.DataFrame(list(dataset.y), columns=['event', 'time'])
        
        merged = dataset.X.reset_index(drop=True).join(y.reset_index(drop=True))
        
        self.model.fit(merged, 'event', 'time')
        
        return self
    
    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival'

    def priorize(self, candidate: Candidate = None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5  # neutral
    
    
    def predict(self, X: pd.DataFrame) -> list[float]:
        """
        Apply prediction model on DataFrame

        Args:
            X (pd.DataFrame): DataFrame use to predict

        Returns:
            list[float]: Predicted values
        """
        results = self.model.predict_cumulative_hazard(X)
        if hasattr(self, 'label_encoder'):
            return self.label_encoder.inverse_transform(results)
        return results
