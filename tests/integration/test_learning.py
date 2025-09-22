import unittest, sys
import pandas as pd
import numpy as np

# Include tools lib
sys.path.append('./src')
from iaml import *
from iaml.explanation import Explanation

class TestLearning(unittest.TestCase):
    
    def __learning_test(self, X, y, shap=True, name=None):
        iaml = IAML(max_duration=60, max_workers=8)
        outputs = iaml.fit(X, y, verbose=-1)
        
        self.assertTrue(isinstance(outputs[0], Candidate), msg=name)
        self.assertTrue(outputs[0].computed_metrics is not None, msg=name)
        
        perf_plots = iaml.chosen_candidate.explain_model_performance(X, y, X_train=X, y_train=y)
        self.assertTrue(isinstance(perf_plots[0], Plot), msg=name)
        
        if shap:
            exp = iaml.chosen_model.explain_model(X.sample(40))
            shap_plots = exp.to_plots()
            self.assertTrue(isinstance(exp, Explanation), msg=name)
            self.assertTrue(isinstance(shap_plots[0], Plot), msg=name)
            
    
    def load_and_sample(self, path, size=1000):
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
        Test : Regression on a CSV file
        """
        datasets = ['./tests/data/insurance.csv',
                    './tests/data/life_expectancy.csv'
                    ]
        
        for dataset in datasets:
            df = self.load_and_sample(dataset)
            y = df['label']
            X = df.drop(columns=['label'])
                
            self.__learning_test(X, y, name=dataset)
    
    
    def test_classification(self):
        """
        Test : Classification on a CSV file
        """
        datasets = ['./tests/data/body_signal_of_smoking.csv',
                    './tests/data/titanic.csv'
                    ]
        
        for dataset in datasets:
            print("DATASET", dataset)
            df = self.load_and_sample(dataset, size=250)
            y = df['label']
            X = df.drop(columns=['label'])
                
            self.__learning_test(X, y, name=dataset)

    def test_survival(self):
        """
        Test : Survival on a CSV file
        """
        datasets = [
                    './tests/data/seer.csv'
                    ]
        
        for dataset in datasets:
            df = self.load_and_sample(dataset)
        
            df['label'] = list(zip(df['event'], df['event_time']))
            df.drop(columns=['event', 'event_time'], inplace=True)
        
            y = df['label']
            X = df.drop(columns=['label'])

            self.__learning_test(X, y, shap=False)
