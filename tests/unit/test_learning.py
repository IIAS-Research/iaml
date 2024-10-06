import unittest, sys
import pandas as pd
import numpy as np

# Include tools lib
sys.path.append('./src/iaml')
from iaml import *

class TestLearning(unittest.TestCase):
    
    def __learning_test(self, X, y):
        iaml = IAML(max_duration=60, max_workers=2)
        outputs = iaml.fit(X, y, verbose=2)

        self.assertTrue(isinstance(outputs[0], Candidate))
        self.assertTrue(outputs[0].computed_metrics is not None)
        

    def test_regression(self):
        """
        Test : Fast regression on a CSV file
        """
        df = pd.read_csv('./src/perf_logger/tests_data/life_expectancy.csv', sep=",")
        labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))
        y = df[labels]
        X = df.drop(columns=labels)
            
        self.__learning_test(X, y)
    
    
    def test_classification(self):
        """
        Test : Fast classification on a CSV file
        """
        df = pd.read_csv('./src/perf_logger/tests_data/fertility.csv', sep=",")
        labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))
        
        y = df[labels]
        X = df.drop(columns=labels)
            
        self.__learning_test(X, y)

    def test_survival(self):
        """
        Test : Fast survival on a CSV file
        """
        df = pd.read_csv('./src/perf_logger/tests_data/seer.csv', sep=",")
        
        df['Status'] = df['Status'].replace({'Alive': False, 'Dead': True})
        df['label'] = list(zip(df['Status'], df['Survival Months']))
        df.drop(columns=['Status', 'Survival Months'], inplace=True)
        
        labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))
        
        y = df[labels]
        X = df.drop(columns=labels)

        self.__learning_test(X, y)
