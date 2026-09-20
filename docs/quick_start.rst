===========
Quick Start
===========

IAML handles preprocessing and model search for you. Start with its defaults
and a small dataset.

Install IAML
============

Use Python 3.10 or newer:

.. code-block:: bash

    python -m pip install PyIAML

Your first pipeline
===================

Save this as ``example.py`` and run ``python example.py``. The dataset is bundled
with scikit-learn, so no data download is needed.

.. code-block:: python

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from iaml import IAML

    if __name__ == "__main__":
        X, y = load_breast_cancer(return_X_y=True, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=42
        )
        search = IAML(max_duration=30, max_workers=1)
        search.fit(X_train, y_train)
        chosen_model = search.chosen_candidate
        print(chosen_model.evaluate(X_test, y_test))

``chosen_model`` is the best model found, including its preprocessing.
``evaluate`` prints its scores on the held-out test set. In this dataset, labels
are ``0`` for malignant and ``1`` for benign.

The example uses one worker and a 30 second search budget. Final fitting can
add time. Keep the ``__main__`` guard when running a script, because training
starts worker processes.

Go further
==========

See :doc:`worked_example` for a complete analysis with performance figures,
SHAP explanations and an exportable record of the selected pipeline.

To understand the choices behind your results and adapt the workflow to your
study, continue with :doc:`usage`, followed by :doc:`evaluation` and
:doc:`explainability`.

The built-in components support the workflow shown here. For study-specific
methods, :doc:`adaptability` explains how to add steps, metrics, validation
splitters and optimizers that reflect your team's research practices.
