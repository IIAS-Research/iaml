import unittest, sys
import pandas as pd
import numpy as np

# Include tools lib
sys.path.append('./src/iaml')
from iaml import *
from iaml.explanation import Explanation

class TestLearningFast(unittest.TestCase):
    
    def __learning_test(self, X, y, shap=True):
        iaml = IAML(max_duration=60, max_workers=8)
        outputs = iaml.fit(X, y, verbose=2)
        
        self.assertTrue(isinstance(outputs[0], Candidate))
        self.assertTrue(outputs[0].computed_metrics is not None)
        
        perf_plots = iaml.chosen_candidate.explain_model_performance(X, y, X_train=X, y_train=y)
        self.assertTrue(isinstance(perf_plots[0], Plot))
        
        if shap:
            exp = iaml.chosen_model.explain_model(X.sample(40))
            shap_plots = exp.to_plots()
            self.assertTrue(isinstance(exp, Explanation))
            self.assertTrue(isinstance(shap_plots[0], Plot))
                

    def load_and_sample(self, path, size=200):
        df = None
        try:
            df = pd.read_csv(path, sep=";")
        except:
            df = pd.DataFrame()
            
        if len(df.columns) < 2:
            df = pd.read_csv(path, sep=",")
            
        if df.shape[0] > size:
            df = df.sample(size)
            
        return df
    
    def test_regression(self):
        """
        Test : Fast regression on a CSV file
        """
        df = self.load_and_sample('./src/perf_logger/tests_data/life_expectancy.csv')
        y = df['label']
        X = df.drop(columns=['label'])
            
        self.__learning_test(X, y)
    
    
    def test_classification(self):
        """
        Test : Fast classification on a CSV file
        """
        df = self.load_and_sample('./src/perf_logger/tests_data/fertility.csv')
        
        y = df['label']
        X = df.drop(columns=['label'])
            
        self.__learning_test(X, y)

    def test_survival(self):
        """
        Test : Fast survival on a CSV file
        """
        df = self.load_and_sample('./src/perf_logger/tests_data/seer.csv')
        
        df['label'] = list(zip(df['event'], df['event_time']))
        df.drop(columns=['event', 'event_time'], inplace=True)
    
        y = df['label']
        X = df.drop(columns=['label'])
    
        self.__learning_test(X, y, shap=False)
