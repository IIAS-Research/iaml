import unittest, sys
import pandas as pd
import numpy as np

# Include tools lib
sys.path.append('./src/iaml')
from iaml import *

class TestAutomed(unittest.TestCase):

    # Test : Fast learning on a CSV file
    def test_fast_learning(self):
        
        df = pd.read_csv('./src/perf_logger/tests_data/life_expectancy.csv', sep=",")
        labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))
        y = df[labels]
        X = df.drop(columns=labels)
            
        automed = IAML(quiet=False, max_duration=30, max_workers=2)
        automed.default_pipeline()
        outputs = automed.fit(X, y)

        self.assertTrue(isinstance(outputs[0], Candidate))
        self.assertTrue(outputs[0].computed_metrics is not None)
