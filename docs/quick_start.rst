==========
QuickStart
==========

Welcome to the IAML QuickStart!
This guide will help you get up and running with IAML in just a few steps. You'll learn how to train a model, evaluate its performance, and interpret the results using IAML's simple and intuitive API.

Installing IAML
===============
First, ensure that IAML is installed. You can install it using pip:

.. code-block:: bash

    pip install iaml

Basic Workflow
--------------
Using IAML to build and evaluate a machine learning model is straightforward. Here's how you can do it:

1. Load your raw dataset.
2. Create an instance of the IAML framework.
3. Train the model on your data.
4. Evaluate the model's performance.
5. Interpret the results.

Minimal Code
============

IAML is designed for ease of use, enabling you to build and evaluate a machine learning model with just a few lines of code.

.. code-block:: python

    from iaml import IAML

    # Load your data (replace with your dataset)
    X_train, X_test, y_train, y_test = ...  # Your training and test datasets

    # Train and evaluate the model
    model = IAML(max_duration=120)  # Optional duration limit
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)

    print("Performance Metrics:", results)

With just a few lines of code:

- The dataset is preprocessed.
- A pipeline is automatically built and optimized.
- Model performance is evaluated using key metrics.

Basic Customization
===================
IAML allows you to customize various parameters to better suit your needs. For example:

.. code-block:: python

    model = IAML(
        main_metric='roc_auc',  # Metric to optimize
        max_workers=4,          # Number of parallel workers
        max_duration=300        # Training duration (in seconds)
    )

    model.fit(X_train, y_train)

You can check the API Reference section to explore all IAML parameters.
For deeper customization, refer to the :ref:`adaptability` page.

Explainability Features
=======================
IAML provides tools to explain model behavior and feature importance. Here's how you can use them:

.. code-block:: python

    model = iaml.chosen_model

    # Generate a summary of the pipeline
    model.explain()

    # Perform detailed feature importance analysis
    model.explain(X_test, y_test)

    # Visualizations (e.g., SHAP plots)
    -> TODO: How to visualize easily?

    # Get scientific references
    model.bibliography()

Full example
============
Below is a basic example of using IAML to solve a classification problem.

.. code-block:: python

    from iaml import IAML
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an IAML instance
    model = IAML(max_duration=120)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    results = model.evaluate(X_test, y_test)
    print("Performance Metrics:", results)

.. note::

    This is an example of a classification task, but IAML automatically adapts to input data. The code remains the same for regression tasks.

What's Next?
============

Explore more advanced features and applications of IAML:

- **Make science with IAML:** Learn how IAML can help you :doc:`create scientific knowledge<scientific>`.
- **Explore all explainability possibilities:** Fully understand your IAML pipeline in the :ref:`explainability` documentation.
- **Preprocessing and Feature Engineering:** Discover how IAML handles data cleaning and transformation automatically in the :ref:`architecture` documentation.
- **Adaptability:** Refer to :ref:`adaptability` to dive deeper into IAML's architecture and learn how to add your own code.

Check out the full documentation for additional details and best practices.
