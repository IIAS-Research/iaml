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
1. **Descriptive Statistics and Visualizations**
    IAML provides tools to generate descriptive statistics and visualizations, enabling researchers to better understand the characteristics of their datasets. These include summary tables, histograms, and correlation matrices that can be directly included in scientific work.

    Example:

    .. code-block:: python

        # TODO

    Visualizations such as histograms, TODO, and TODO are automatically generated to provide a clear overview of the data.

2. **Detailed Documentation of Pipelines**
    IAML generates summaries for every pipeline, documenting preprocessing steps, feature engineering techniques, and models used. This ensures that all methods are transparent and can be cited or replicated in scientific papers.

    Example:

    .. code-block:: python

        model.explain()  # Summarizes the entire pipeline, including transformations and models.

3. **Comprehensive Performance Metrics**
    Evaluate models with multiple metrics to provide a detailed view of performance. These metrics can be directly used in publications.

    Example:

    .. code-block:: python

        results = model.evaluate(X_test, y_test)
        print("Performance Metrics:", results)

        # Example output:
        # {'accuracy': 0.85, 'precision': 0.88, 'recall': 0.83, 'f1_score': 0.86}

4. **Feature Importance Analysis**
    IAML integrates SHAP (SHapley Additive Explanations) to provide detailed feature importance analysis, helping you interpret your model and draw meaningful conclusions from your data.

    Example:

    .. code-block:: python

        model.explain(X_test, y_test)  # Provides SHAP-based insights.

5. **Model Performance Visualizations**
    IAML automatically generates visualizations to help researchers understand model performance, strengths, and weaknesses. These include:

    - ROC curves for classification tasks.
    - Residual plots for regression tasks.
    - Confusion matrices to analyze prediction errors.

    Example:

    .. code-block:: python

        TODO

6. **Scientific Bibliography**
    Automatically generate a bibliography of all algorithms and methods used in the pipeline, making it easier to include proper citations in your work.

    Example:

    .. code-block:: python

        print(model.bibliography())

        # Example output:
        # TODO

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
