==================
IAML for Science
==================

IAML is designed to empower researchers and practitioners in creating high-quality scientific work. Its modularity, transparency, and explainability make it an excellent companion for data-driven research, enabling reproducibility, insights, and efficient analysis. This page explores how IAML can assist in your scientific endeavors.

Why Use IAML for Scientific Research?
======================================
When conducting scientific research, it is crucial to ensure that the methodology is robust, reproducible, and interpretable. IAML provides several features tailored to the needs of researchers:

- **Reproducibility:** IAML ensures that every pipeline and process can be replicated, with all parameters and transformations documented.
- **Explainability:** Gain insights into model behavior, feature importance, and decision-making processes using integrated tools like SHAP.
- **Efficiency:** Automate time-consuming tasks such as model selection, hyperparameter tuning, and preprocessing, allowing you to focus on interpreting results.
- **Customizability:** Adapt pipelines and integrate domain-specific methods to meet the unique requirements of your research.
- **Performance:** Achieve state-of-the-art results with minimal effort, ensuring your findings are backed by reliable models and analysis.

Key Features for Scientific Work
================================
.. 1. **Descriptive Statistics and Visualizations**
..     IAML provides tools to generate descriptive statistics and visualizations, enabling researchers to better understand the characteristics of their datasets. These include summary tables, histograms, and correlation matrices that can be directly included in scientific work.

..     Example:

..     .. code-block:: python

..         # TODO

..     Visualizations such as histograms, TODO, and TODO are automatically generated to provide a clear overview of the data.

1. **Detailed Documentation of Pipelines**
    IAML generates summaries for every pipeline, documenting preprocessing steps, feature engineering techniques, and models used. This ensures that all methods are transparent and can be cited or replicated in scientific papers.

    Example:

    .. code-block:: python

        model.describe_steps()  # Summarizes the entire pipeline, including transformations and models.

2. **Comprehensive Performance Metrics**
    Evaluate models with multiple metrics to provide a detailed view of performance. These metrics can be directly used in publications.

    Example:

    .. code-block:: python

        results = model.evaluate(X_test, y_test)
        print("Performance Metrics:", results)

        # Example output:
        # {'accuracy': 0.85, 'precision': 0.88, 'recall': 0.83, 'f1_score': 0.86}

3. **Feature Importance Analysis**
    IAML integrates SHAP (SHapley Additive Explanations) to provide detailed feature importance analysis, helping you interpret your model and draw meaningful conclusions from your data.

    Example:

    .. code-block:: python

        explanation = model.explain_feature_importance(X_test, y_test)  # Provides SHAP-based insights.

        # Example output:
        # {'Pclass': 0.09936865575659631,
        #  'Name': 0.18412368391698725,
        #  'Sex': 0.13135919778281502,
        #  'Age': 0.017263125201303287,
        #  'SibSp': 0.01746738519962855,
        #  'Parch': 0.011230812036885408,
        #  'Ticket': 0.02003346312480449,
        #  'Fare': 0.008829174755618716,
        #  'Cabin': 0.005968384526857956,
        #  'Embarked': 0.032069830830315095}

4. **Model Performance Visualizations**
    IAML automatically generates visualizations to help researchers understand model performance, strengths, and weaknesses. These include:

    - ROC curves for classification tasks.
    - Residual plots for regression tasks.
    - Confusion matrices to analyze prediction errors.

    Example:

    .. code-block:: python

        plots = model.explain_model_performance(X_test, y_test)  # Provides SHAP-based insights.

        # Example output:
        # [<iaml.plots.class_prediction_error_plot.ClassPredictionErrorPlot object at 0x7f7d884260b0>,
        #  <iaml.plots.classification_report_plot.ClassificationReportPlot object at 0x7f7da5422380>,
        #  <iaml.plots.confusion_matrix_plot.ConfusionMatrixPlot object at 0x7f7d7db05f30>,
        #  <iaml.plots.rocauc_plot.ROCAUCPlot object at 0x7f7d7da0f880>,
        #  <iaml.plots.precision_recall_curve_plot.PrecisionRecallCurvePlot object at 0x7f7d88247bb0>]

        # Example usage in a Jupyter notebook:
        from IPython.display import Image

        for plot in plots:
            display(Image(plot.image))

5. **Scientific Bibliography**
    Automatically generate a bibliography of all algorithms and methods used in the pipeline, making it easier to include proper citations in your work.

    Example:

    .. code-block:: python

        print(model.bibliography())

        # Example output:
        # [ 1]  G. E. P. Box, D. R. Cox. An Analysis of Transformations
        # Journal of the Royal Statistical Society: Series B (Methodological),                 Vol.26, No.2 page 211--243, https://doi.org/10.1111/j.2517-6161.1964.tb00553.x, 1964.
        # [ 2]  In-Kwon Yeo, Richard A. Johnson. A New Family of Power Transformations to Improve Normality or Symmetry
        # Oxford University Press, Biometrika Vol.87 No.4 page 954--959, https://doi.org/10.1093/biomet/87.4.954, 2000.
        # [ 3]  Joseph Berkson. Application of the Logistic Function to Bio-Essay
        # Journal of the American Statistical Association Vol. 39, No. 227, page 357--365, https://doi.org/10.2307/2280041, 1944.

How to Cite IAML
=================

If you use IAML in your work, please consider citing it. Below is the recommended citation format:

.. code-block::

    Merieux, R., Ruellet, H., Bourachot, R., Dahlouk, Y. A., Blanchard, F., Vuiblet, V. 
    "IAML: A Modular and Explainable AutoML Framework for High-Performance on Tabular Data."
    Preprint submitted to Knowledge-Based Systems, 2024.


Conclusion
==========

IAML is designed to bridge the gap between automated machine learning and rigorous scientific research. By combining performance, transparency, and reproducibility, IAML enables researchers to focus on generating insights and advancing knowledge while minimizing the effort spent on technical implementation.

Get started today and let IAML power your next scientific breakthrough!
