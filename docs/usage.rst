======
Usage
======

This guide explains, with more details than :doc:`quick_start`, how to use IAML for training, predicting, and interpreting machine learning models. 

Setup
=====

From the repository root, install IAML in your Python environment:

.. code-block:: bash

    python -m pip install .

How to Create and Train a Model
===============================

Creating and training a model with IAML is straightforward. Follow these steps:

1. Import the IAML class.
2. Create an instance of the IAML class.
3. Train the model on your data using `fit`.

Example:

.. code-block:: python

    from iaml import AccuracyMetric, IAML

    # Create an IAML instance
    search = IAML(max_duration=120, main_metric=AccuracyMetric(), max_workers=1)

    # Train the model
    candidates = search.fit(X_train, y_train)
    model = candidates[0]

    # Output: Trained pipeline ready for evaluation and prediction.

The remaining examples use ``model`` for the trained
:py:class:`~iaml.candidate.Candidate` returned by ``fit``. Features and targets
are pandas DataFrames; convert a target Series with ``y.to_frame()``. In a
Python script, place training inside an ``if __name__ == "__main__":`` guard
as shown in :doc:`quick_start`.

IAML Class Parameters
---------------------
As shown in the example, when creating an instance of the `IAML` class, you can configure various parameters to control its behavior.
Below is a detailed explanation of the most importants parameters. Please consider using the API reference to learn the other parameters.

- **max_workers (int, optional):**  
    Specifies the maximum number of parallel workers to use during training.  

    - **Default:** Number of available CPU cores.
    - **Use Case:** Set this to control resource usage in multi-core systems.

- **max_stage_duration (int, optional):**  
    Maximum time (in seconds) allocated to each stage of the training process. If not set, it defaults to `max_duration / 5` or 900 seconds, whichever is higher.  
    
    - **Default:** ``max(max_duration / 5, 900)``.
    - **Use Case:** Fine-tune the allocation of time for different pipeline stages.

- **max_duration (int, optional):**  
    The search time budget in seconds, including candidate submission and evaluation.
    Initialization and final fitting can take additional time.

    - **Default:** ``-1`` (no limit).  
    - **Use Case:** Limit the overall training time for faster iterations or resource constraints.

- **main_metric (Metric, optional):**  
    The primary metric to optimize during training (e.g., accuracy, ROC AUC).

    - **Default:** ``None`` (selected from the task type).
    - **Use Case:** Pass a metric instance, such as ``AccuracyMetric()`` for classification or ``R2ScoreMetric()`` for regression. The defaults are balanced accuracy for classification, R² for regression, and IPCW concordance for survival.


Fit Parameters
--------------
The :py:meth:`~iaml.iaml.IAML.fit` method trains pipeline and model on your data. Below are the most useful parameters. Please consider using the API reference to learn the other parameters.

- **X (pd.DataFrame):**  
    The input features for training.  

    - **Required.**
    - **Type:** A pandas DataFrame containing the training data.

- **y (pd.DataFrame):**  
    The target labels for training.  

    - **Required.**
    - **Type:** A pandas DataFrame containing the labels corresponding to ``X``.

- **groups (pd.DataFrame, optional):**  
    Group labels for the samples, used for group-aware cross-validation. 

    - **Default:** ``None``.  
    - **Use Case:** Use when working with grouped datasets where samples should not be split across folds.

- **groups_columns (list[str], optional):**  
    The columns in the ``X`` dataset that define groups. If specified, ``groups`` is inferred from these columns.  

    - **Default:** ``None``.

- **patience (int, optional):**  
    Number of generations without improvement before training stops.  

    - **Default:** ``-1`` (no early stopping based on patience).  
    - **Use Case:** Set to a positive integer to control convergence and prevent unnecessary iterations.

- **verbose (int, optional):**  
    Controls the level of logging output during training.  

    - **Default:** ``1`` (minimal logs).  
    - **Options:**  
        - ``0``: Silent mode.  
        - ``1``: Minimal logging.  
        - ``2``: Detailed logging.  

What to Do After Training a Pipeline
=====================================
Once a pipeline is trained, you can:

- **Evaluate the Model:** Use the :py:meth:`~iaml.candidate.Candidate.evaluate` method to assess the model's performance.
- **Make predictions:** Use the model to predict labels from samples
- **Generate Explanations:** Use built-in explainability tools to interpret model behavior.

Evaluate the Model
------------------
Once the model is trained, you can evaluate its performance on your own test dataset, if you have one. The :py:meth:`~iaml.candidate.Candidate.evaluate` method allows you to assess the model's performance based on several metrics implemented in IAML, on a dataset that IAML has not seen before.

.. code-block:: python

    # Evaluate the model
    metrics = model.evaluate(X_test, y_test)

    # Print the results
    print('Performance metrics:')
    for metric, value in metrics.items():
        print(' ', metric, '=', value)

    # Performance metrics:
    #   accuracy = 0.8324022346368715
    #   balanced_accuracy = 0.8174492914698528
    #   classification_error = 0.1825507085301472
    #   f1_score = 0.8717948717948718
    #   precision = 0.8793103448275862
    #   recall = 0.864406779661017
    #   specificity = 0.864406779661017
    #   ROC AUC = 0.883023061961656


Positive Class for Binary Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``PrecisionMetric``, ``RecallMetric`` and ``F1ScoreMetric`` use a fixed convention
for the positive class, independent of row order:

- For labels ``0/1``, ``False/True`` or ``-1/1``, the positive class is ``1``.
- For other binary labels, it is the last label in sorted order (numerical or
  alphabetical).
- Candidate evaluation uses the training labels as a reference, including when
  the test set contains only one class.

For a direct metric calculation, choose a different positive class explicitly:

.. code-block:: python

    from iaml import PrecisionMetric, RecallMetric, F1ScoreMetric

    metrics = [
        PrecisionMetric(pos_label="case"),
        RecallMetric(pos_label="case"),
        F1ScoreMetric(pos_label="case"),
    ]
    scores = {str(metric): metric.compute(y_test, y_pred) for metric in metrics}

When only one nonstandard label is available, automatic selection is ambiguous:
provide ``pos_label`` or pass ``y_train`` containing both classes to ``compute``.
Undefined binary scores return zero. Multiclass targets use weighted averaging,
including when a test subset is missing classes present in the training data.
Multilabel targets use sample averaging. Historical binary scores may differ
because older versions used the first observation's label as the positive class.


Make Predictions
-----------------------
Once a model is trained, use the :py:meth:`~iaml.candidate.Candidate.predict` or :py:meth:`~iaml.candidate.Candidate.predict_proba` method to generate predictions on new data.

Example:

.. code-block:: python

    # Generate predictions
    predictions = model.predict(X_test)

    # Output: Array of predicted values

Generate Explanations
============================
IAML provides tools to explain model decisions and behavior. These include feature importance analysis, pipeline summaries, and performance plots. The :py:meth:`~iaml.candidate.Candidate.explain_feature_importance` returns an :py:class:`~iaml.explanation.Explanation` object which you can use to extract SHAP values and plots. You can learn more about :doc:`explainability` on its own page.

Example:

.. code-block:: python

    # Perform detailed feature importance analysis
    explanation = model.explain_feature_importance(X_test)

    # Generate a Markdown table of the feature importances
    explanation.to_markdown_shap()

    # Generate SHAP plots
    explanation.to_markdown_plots()
