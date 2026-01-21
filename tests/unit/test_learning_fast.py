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
                

    def test_regression(self):
        """
        Test : Fast regression on synthetic data
        """
        X, y = make_regression_data(n_samples=80, seed=110)
        self.__learning_test(X, y)
    
    
    def test_classification(self):
        """
        Test : Fast classification on synthetic data
        """
        X, y = make_classification_data(n_samples=80, seed=120)
        self.__learning_test(X, y)

    def test_survival(self):
        """
        Test : Fast survival on synthetic data
        """
        X, y = make_survival_data(n_samples=80, seed=130)
        self.__learning_test(X, y, shap=False)
