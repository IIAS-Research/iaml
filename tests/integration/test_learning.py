import unittest, sys

# Include tools lib
sys.path.append('./src')
from iaml.iaml import IAML
from iaml.candidate import Candidate
from iaml.plot import Plot
from iaml.explanation import Explanation

from tests.helpers.datasets import (
    make_classification_data,
    make_regression_data,
    make_survival_data,
)

class TestLearning(unittest.TestCase):
    
    def __learning_test(self, X, y, shap=True, name=None):
        iaml = IAML(max_duration=60, max_workers=1)
        outputs = iaml.fit(X, y, verbose=-1)
        
        self.assertTrue(isinstance(outputs[0], Candidate), msg=name)
        self.assertTrue(outputs[0].computed_metrics is not None, msg=name)
        
        perf_plots = iaml.chosen_candidate.explain_model_performance(X, y, X_train=X, y_train=y)
        self.assertTrue(isinstance(perf_plots[0], Plot), msg=name)
        
        if shap:
            sample_size = min(40, len(X))
            exp = iaml.chosen_model.explain_model(X.sample(sample_size))
            shap_plots = exp.to_plots()
            self.assertTrue(isinstance(exp, Explanation), msg=name)
            self.assertTrue(isinstance(shap_plots[0], Plot), msg=name)
            
    def test_regression(self):
        """
        Test : Regression on synthetic data
        """
        X, y = make_regression_data(n_samples=30, seed=10)
        self.__learning_test(X, y, name="synthetic_regression")
    
    
    def test_classification(self):
        """
        Test : Classification on synthetic data
        """
        X, y = make_classification_data(n_samples=30, seed=20)
        self.__learning_test(X, y, name="synthetic_classification")

    def test_survival(self):
        """
        Test : Survival on synthetic data
        """
        X, y = make_survival_data(n_samples=30, seed=30)
        self.__learning_test(X, y, shap=False, name="synthetic_survival")
