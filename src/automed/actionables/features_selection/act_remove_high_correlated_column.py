"""
[STEP] Remove High Correlated Column
"""
import numpy as np
import pandas as pd
from ...actionable import Actionable
from ...output import Output, Input
from ...step import is_step, runner

@is_step('features_selection')
class ActRemoveHighCorrelatedColumn(Actionable):
    """
    [STEP] Remove High Correlated Column
    """
    name = 'Remove High Correlated Column'
    description = 'Remove columns which correlation with other columns is higher than {threshold}.'

    def __init__(self):
        self.configurations = [{
            'threshold': {
                'description': 'If two columns is correlated over this value, only one \
                    will be kept',
                'default': 0.9
            }
        }]
        self.to_drop:list[str] = None
    
    @runner
    def run(self, input_data: Input, callback:callable=None) -> Output: # pylint: disable=unused-argument
        """
        Find high correlated columns to drop

        Args:
            input_data (Input): Fit data
            callback (callable, optional): Call after each step. Defaults to None.

        Returns:
            Output: Transformed input
        """
        # Compute correlation matrix 
        corr_matrix = input_data.dataset.X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        
        # Find features with above-threshold correlation
        # self.to_drop = list(to_drop.keys())
        corr = { c: (upper[upper[c] >= self.get_config('threshold')].index) for c in upper.columns }
        self.to_drop = [ c for c, v in corr.items() if len(v) > 0 ]

        input_data.pipeline.add_explanation(self, [
            f"""Dropped column **`{c}`** because it was too correlated with
                {", ".join([ f"**`{i}`**" for i in corr[c] ])}."""
            for c in self.to_drop
        ])
        
        return input_data.add_transform(self)
    
    
    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
        """
        Drop high correlated column

        Args:
            x (pd.DataFrame): DataFrame to transform

        Returns:
            pd.DataFrame: Transformed dataset
        """
        return X.drop(self.to_drop, axis=1)

        
    
    def priorize(self, input_data:Input=None) -> float:
        """
        Try to priorize himself

        Return : continuous between 0 and 1
        """
        return 0.5
