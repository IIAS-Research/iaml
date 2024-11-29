"""
[STEP] Remove Low Variance Column
"""
import textwrap
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
from ...actionable import Actionable
from ...dataset import Dataset
from ...candidate import Candidate
from ...decorators.all import is_step

@is_step('features_selection')
class ActRemoveLowVarianceColumn(Actionable):
    """
    [STEP] Remove Low Variance Column
    """
    name = 'Remove Low Variance Column'
    description = 'Remove columns with variance lower than {threshold}.'
    description_long = textwrap.dedent('''\
        Remove features from the dataset that have variance lower than
        the specified threshold. Low variance columns do not contribute 
        significantly to the predictive power of models and can lead to 
        overfitting.''')

    def __init__(self):
        self.configuration = {
            'threshold': {
                'description': 'Columns with variance lower than this value will be dropped.',
                'default': 1e-10
            }
        }
        self.selector: VarianceThreshold = None
        self.to_drop: list[str] = None

    def fit(self, dataset: Dataset) -> Actionable:
        """
        Identify low variance columns to drop.

        Args:
            dataset (Dataset): Input dataset for analysis

        Returns:
            Actionable: Self (for chaining)
        """
        threshold_value = self.get_config('threshold')
        
        # Identify the columns that are being dropped (features with low variance)
        self.to_drop = self.__get_columns(dataset)
        
        # Create explanations for each dropped feature
        self.explanations = [
            f"""Dropped column **`{col}`** because its variance was too low \
                (below threshold {threshold_value})."""
            for col in self.to_drop
        ]
        
        return self
    
    def __get_columns(self, dataset: Dataset) -> list:
        # Set up VarianceThreshold selector with the user-defined threshold
        threshold_value = self.get_config('threshold')
        selector = VarianceThreshold(threshold=threshold_value)
        
        # Apply selector to dataset to identify features to keep
        selector.fit(dataset.X)

        # Get the boolean mask of features to keep (features with sufficient variance)
        feature_mask = selector.get_support()

        # Identify the columns that are being dropped (features with low variance)
        return list(dataset.X.columns[~feature_mask])
        

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Drop low variance columns from the DataFrame.

        Args:
            X (pd.DataFrame): The dataset to transform

        Returns:
            pd.DataFrame: Transformed dataset without low variance columns
        """
        return X.drop(columns=self.to_drop)

    def priorize(self, candidate: Candidate = None) -> float:
        """
        Try to priorize itself.

        Returns:
            float: A neutral score (0.5)
        """
        return 0.5
    
    def suitable(self, dataset: Dataset) -> bool:
        return dataset.type_of_target == 'survival' and self.__get_columns(dataset)
    
