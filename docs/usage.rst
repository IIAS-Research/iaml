======
Usage
======

This guide explains, with more details than :doc:'Quick Start<quick_start>', how to use IAML for training, predicting, and interpreting machine learning models. 

Setup
=====

Before diving into the specifics, ensure that you have IAML installed:

.. code-block:: bash

    pip install iaml

How to Create and Train a Model
===============================

Creating and training a model with IAML is straightforward. Follow these steps:

1. **Import the IAML class.**
2. **Create an instance of the IAML class.**
3. **Train the model on your data using `fit`.**

Example:

.. code-block:: python

    from iaml import IAML

    # Create an IAML instance
    model = IAML(max_duration=120, main_metric='accuracy')

    # Train the model
    model.fit(X_train, y_train)

    # Output: Trained pipeline ready for evaluation and prediction.

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
    
    - **Default:** `max(max_duration / 5, 900)`.
    - **Use Case:** Fine-tune the allocation of time for different pipeline stages.

- **max_duration (int, optional):**  
    The total time (in seconds) allocated for the entire training process.

    - **Default:** `-1` (no limit).  
    - **Use Case:** Limit the overall training time for faster iterations or resource constraints.

- **main_metric (Metric, optional):**  
    The primary metric to optimize during training (e.g., accuracy, ROC AUC).

    - **Default:** `None` (must be specified).  
    - **Use Case:** Choose the most relevant metric for your task, such as `'accuracy'` for classification or `'r2'` for regression.


Fit Parameters
--------------
The `fit` method trains pipeline and model on your data. Below are the most useful parameters. Please consider using the API reference to learn the other parameters.

- **X (pd.DataFrame):**  
    The input features for training.  

    - **Required.**
    - **Type:** A pandas DataFrame containing the training data.

- **y (pd.DataFrame):**  
    The target labels for training.  

    - **Required.**
    - **Type:** A pandas DataFrame containing the labels corresponding to `X`.

- **groups (pd.DataFrame, optional):**  
    Group labels for the samples, used for group-aware cross-validation. 

    - **Default:** `None`.  
    - **Use Case:** Use when working with grouped datasets where samples should not be split across folds.

- **groups_columns (list[str], optional):**  
    The columns in the `X` dataset that define groups. If specified, `groups` is inferred from these columns.  

    - **Default:** `None`.

- **patience (int, optional):**  
    Number of generations without improvement before training stops.  

    - **Default:** `-1` (no early stopping based on patience).  
    - **Use Case:** Set to a positive integer to control convergence and prevent unnecessary iterations.

- **verbose (int, optional):**  
    Controls the level of logging output during training.  

    - **Default:** `1` (minimal logs).  
    - **Options:**  
        - `0`: Silent mode.  
        - `1`: Minimal logging.  
        - `2`: Detailed logging.  

What to Do After Training a Pipeline
=====================================
Once a pipeline is trained, you can:

- **Evaluate the Model:** Use the `evaluate` method to assess the model's performance.
- **Make predictions:** Use the model to predict labels from samples
- **Generate Explanations:** Use built-in explainability tools to interpret model behavior.

Evaluate the Model
------------------
TODO texte ici

.. code-block:: python

    # Evaluate the model
    results = model.evaluate(X_test, y_test)
    print("Performance Metrics:", results)

    # TODO output exemple


Make Predictions
-----------------------
Once a model is trained, use the `predict` or `predict_proba` method to generate predictions on new data.

Example:

.. code-block:: python

    # Generate predictions
    predictions = model.predict(X_test)

    # Output: Array of predicted values

Generate Explanations
============================
IAML provides tools to explain model decisions and behavior. These include feature importance analysis, pipeline summaries, and performance plots.

TODO