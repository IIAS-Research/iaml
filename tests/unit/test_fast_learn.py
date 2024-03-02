import unittest, sys
import pandas as pd
import numpy as np

# Include tools lib
sys.path.append('./src/')
from automed import *

class TestAutomed(unittest.TestCase):

    # Test : Fast learning on a CSV file
    def test_fast_learning(self):
        
        df = pd.read_csv('./src/perf_logger/tests_data/life_expectancy.csv', sep=",")
        dataset = Dataset(df)
        labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))
        dataset.set_label(labels)
            
        automed = AutoMed(dataset, quiet=False)
        automed.debug_load(fast=True)
        outputs = automed.run()

        self.assertTrue(isinstance(outputs[0], Output))