# from ..actionable.automed import *
import sys
sys.path.insert(0, '../python_project/src/')
from automed.automed import *

from pandas.api.types import is_string_dtype
import re
import pandas as pd

@is_step('cleaning')
class ActAtcCode(Actionable):
    name = "Features engineering on ATC codes"
    configuration = {}
    
    main_categories = {
        'A': 'alimentary',
        'B': 'blood',
        'C': 'cardiovascular',
        'D': 'dermatological',
        'G': 'genitourinary',
        'H': 'hormonal',
        'J': 'antiinfective',
        'L': 'antineoplastic',
        'M': 'musculoskeletal',
        'N': 'nervous',
        'P': 'antiparasitic',
        'R': 'respiratory',
        'S': 'sensor',
        'V': 'various'
    }
    
    @runner
    def run(self, input, callback=None) -> Output:
        columns_to_add = {'test': pd.DataFrame(), 'train': pd.DataFrame()}
        for column in input.dataset.X_train.columns:
            values = {'test': input.dataset.X_test[column], 'train': input.dataset.X_train[column]}
            if is_string_dtype(values['train']) and self._is_atc_column(values['train']):
                for env in ['test', 'train']:
                    columns_to_add[env] = pd.concat([columns_to_add[env], self.transform(column, values[env])], axis=1)
    
        input.dataset.X_train = pd.concat([input.dataset.X_train, columns_to_add['train']], axis=1)
        input.dataset.X_test = pd.concat([input.dataset.X_test, columns_to_add['test']], axis=1)
        
        return input.to_output(input.dataset, None, None)
    
    def priorize(self, input=None):
        return 0.9
    
    def transform(self, column_name, values):
        # Main categories
        def to_apply(value):
            if value:
                return self.main_categories[value[0]]
            else:
                return None
        
        return pd.DataFrame({f"{column_name}_atc_category": values.apply(to_apply)})
    
    def _is_atc_column(self, column):
        # TODO -> test all values
        return self._is_atc(column[0])
    
    def _is_atc(self, code):
        return bool(re.fullmatch(r"(A|B|C|D|G|H|J|L|M|N|P|R|S|V)\d{2}[\w|\d]*", code))