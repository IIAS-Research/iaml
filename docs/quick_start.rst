===========
Quick Start
===========

Train and evaluate your first model with IAML's built-in pipeline. This
walkthrough uses the defaults for preprocessing, model selection and
cross-validation; no pipeline configuration is needed.

Install IAML
============

Use Python 3.10 or newer:

.. code-block:: bash

    python -m pip install PyIAML

Your first study
================

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
        print(search.get_descriptive_statistics(X_train, y_train))
        search.fit(X_train, y_train)
        chosen_model = search.chosen_candidate
        print(chosen_model.evaluate(X_test, y_test))

The descriptive table is calculated before training. ``chosen_model`` is the
best model found, including its preprocessing; ``evaluate`` returns its scores
on the held-out test set. In this dataset, labels are ``0`` for malignant and
``1`` for benign.

The example uses one worker and a 30 second search budget. Final fitting can
add time. Keep the ``__main__`` guard when running a script, because training
starts worker processes.

Continue the integrated workflow
================================

See :doc:`worked_example` for a complete analysis with performance figures,
SHAP explanations and an exportable record of the selected pipeline.

The guides follow the same study from training through reporting:

* :doc:`usage`: set the task, validation design, search budget and fitting data.
* :doc:`evaluation`: assess held-out predictions and create performance figures.
* :doc:`explainability`: inspect the selected pipeline and explain predictions
  with SHAP.
* :doc:`scientific`: retain configurations, results and method references for
  your study report.

Descriptive statistics, performance plots and explanations are calculated when
you request them; fitting does not run every available analysis.

Compose pipelines for your team
===============================

For reusable recipes and study-specific methods, continue with
:doc:`discover_pipelines`. This optional tour introduces the
:doc:`Pipeline API <pipelines/index>` and :doc:`component extensions <adaptability>`
while keeping the training and evaluation workflow shown here.
