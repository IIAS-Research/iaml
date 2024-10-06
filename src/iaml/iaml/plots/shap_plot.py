"""
[PLOT] Wrap Shap Plot 
"""
from functools import wraps
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import textwrap
import shap
import io


from ..plot import Plot

class ShapPlot(Plot):
    """
    [PLOT] Wrap Shap Plot 
    """
    def __init__(self, plot_key, shaps_values, *args, ps=None, scatter_feature=None, **kwargs):
        if ps is None:
            ps = slice(0, len(shaps_values))
            
        if scatter_feature is None and plot_key == 'scatter':
            scatter_feature = shaps_values.feature_names[0]
            
        method, self.title, self.description = self.__plots_informations(plot_key, shaps_values)
        self._binary_image = io.BytesIO()
        
        if scatter_feature:
            method(shaps_values[ps, scatter_feature], *args, show=False, **kwargs)
        else:
            method(shaps_values[ps.start], *args, show=False, **kwargs)
            
        
        plt.savefig(self._binary_image, bbox_inches='tight')
        # self._binary_image.seek(0)
        plt.close()
        
    @classmethod
    def all(cls, shaps_values, *args, **kwargs) -> list:
        """
        Get instance for all kind of Shap Plot
        """
        plots = []
        for key in ['force', 'waterfall', 'beeswarm', 'scatter', 'heatmap', 'bar']:
            plots.append(cls(key, shaps_values, *args, **kwargs))
            
        return plots
    
    def __plots_informations(self, key:str, shap_values) -> tuple[str]:
        """
        Returns the title and description for various SHAP plot types in simple terms.

        This function provides easy-to-understand explanations for SHAP plots, using 
        examples from the medical field, to help non-experts interpret how machine learning 
        models make predictions.

        Parameters:
        -----------
        key : str
            The type of SHAP plot ('force', 'waterfall', 'beeswarm', 'scatter', 'heatmap', 'bar').

        Returns:
        --------
        tuple[str]:
            A title and description of the SHAP plot type, explaining what it shows and how it 
            relates to model predictions.
        """
        features = shap_values.feature_names
        values = shap_values[0].values
        
        match key:
            case 'force':
                force_shap, force_feature = max(zip(values, features), key=lambda v: abs(v[0]))
                return (shap.plots.force,
                    "SHAP Force Plot: Visualizing How Individual Factors Contribute \
                    to a Prediction",
                    textwrap.dedent(f"""
                    The force plot shows how different factors (e.g., age, cholesterol level, blood 
                    pressure) push the model’s prediction for an individual patient. It explains 
                    whether each factor increases or decreases the likelihood of a certain outcome, 
                    such as a diagnosis of heart disease. Red arrows indicate factors increasing risk, 
                    while blue arrows show those reducing risk. This plot helps interpret the specific 
                    impact of each factor for a given prediction.
                    
                    Reading: For this prediction, `{force_feature}` impacts the final prediction 
                    value by {force_shap:.3f}.
                    """))
            case 'waterfall':
                force_shap, force_feature = max(zip(values, features), key=lambda v: abs(v[0]))
                return (shap.plots.waterfall,
                    "SHAP Waterfall Plot: Decomposing a Prediction into Its Components",
                    textwrap.dedent(f"""
                    The waterfall plot breaks down how each factor influences a single patient's 
                    prediction by showing the cumulative effect of each factor. Starting from the 
                    average prediction, it steps through each factor (e.g., age, medication history, 
                    lab results) to show how the final prediction is reached. This helps in 
                    understanding the main contributors to a prediction, such as a high blood 
                    sugar level increasing the risk of diabetes.
                    
                    Reading: For this prediction, `{force_feature}` impacts the final prediction 
                    value by {force_shap:.3f}.
                    """))
            case 'beeswarm':
                return (shap.plots.beeswarm,
                    "SHAP Beeswarm Plot: Identifying the Most Important Factors \
                    Across All Patients",
                    textwrap.dedent("""
                    The beeswarm plot highlights which factors are most important across all patients.
                    Each dot represents a patient, with dots positioned based on the factor's impact 
                    on the prediction (e.g., positive or negative impact on disease risk). For instance, 
                    a cluster of red dots could show that high blood pressure consistently increases 
                    heart disease risk. This plot helps find patterns and common trends in the data.
                    """))
            case 'scatter':
                return (shap.plots.scatter,
                    "SHAP Scatter Plot: Visualizing the Relationship Between a Factor \
                    and Prediction",
                    textwrap.dedent("""
                    The scatter plot shows the relationship between a specific factor (e.g., body mass index)
                    and its SHAP value, which tells us how much it affects the model’s prediction. 
                    By plotting multiple patients, this plot reveals how changes in a factor (like increasing
                    BMI) can lead to higher or lower risk predictions (such as for heart disease).
                    """))
            case 'heatmap':
                return (shap.plots.heatmap,
                    "SHAP Heatmap: Understanding Factor Importance Across Multiple Patients",
                    textwrap.dedent("""
                    The heatmap shows the impact of different factors for many patients, with color 
                    intensity representing how strongly a factor influences the model's prediction. 
                    For example, dark red may highlight that high cholesterol is a strong positive 
                    predictor for heart disease in several patients. This plot helps you see which 
                    factors are the most influential overall.
                    """))
            case 'bar':
                mean_shap = np.abs(shap_values.values).mean(axis=0)
                bar_shap, bar_feature = max(zip(mean_shap, features), key=lambda v: v[0])
                return (shap.plots.bar,
                    "SHAP Bar Plot: Ranking the Most Important Factors",
                    textwrap.dedent(f"""
                    The bar plot ranks the factors by their overall importance in the model’s 
                    predictions. Each bar represents a factor (e.g., age, smoking status, 
                    cholesterol level) and shows how much it contributed to the model’s 
                    decision-making process across all patients. This helps identify the key 
                    factors driving predictions, such as high blood pressure being the most 
                    influential predictor of heart disease.
                    
                    Reading: `{bar_feature}` has an absolute impact of {bar_shap:.3f}
                    on the average final prediction value.
                    """))
