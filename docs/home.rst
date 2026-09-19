=======================================
Integrated AutoML for Medical Labs
=======================================

**IAML (Integrated AutoML for Medical Labs)** is a Python framework for clinical
research with tabular data. It brings preprocessing, model search and evaluation
into one workflow, with tools to inspect the resulting pipelines and explain
their predictions.

Clinical research workflow
==========================

IAML helps researchers build prediction models for classification, regression
and survival analysis through a Python API. Clinical researchers and data
scientists can review the selected methods, evaluate predictions on held-out
data and use explanations to discuss model behavior.

Each pipeline is assembled from modular steps for data preparation and modeling.
IAML uses genetic search by default; alternative optimizers include Bayesian
hyperparameter tuning. Researchers can choose evaluation metrics and validation
settings appropriate to their study, then inspect the trained candidates.

Key Features
============

- **Accessible workflow:** Build and compare pipelines through a common Python API.
- **Inspectable methods:** Review pipeline steps, their parameters and evaluation metrics.
- **Explanations:** Request SHAP feature explanations and task-specific performance plots.
- **Study reporting:** Collect method references and optionally retain cross-validation records.
- **Modularity:** Configure or extend components to suit the data and research question.

Basic example
=============
Given training data and held-out features, search for a pipeline and use the
selected candidate to make predictions:

.. code-block:: python

   from iaml import IAML

   if __name__ == "__main__":
       search = IAML(max_duration=30, max_workers=1)
       candidates = search.fit(X_train, y_train)
       predictions = candidates[0].predict(X_test)

See :doc:`quick_start` for a complete example with pandas DataFrames.
For more advanced usage, such as custom metrics or explainability features, refer to the :doc:`usage` section.

Documentation guides
====================

- :doc:`quick_start`: install IAML and run a complete example.
- :doc:`architecture`: understand pipeline construction and optimization.
- :doc:`scientific`: inspect and report methods and record experiment settings.
- :doc:`explainability`: review predictions and feature explanations.
- :doc:`adaptability`: configure components and add study-specific methods.
- :doc:`references`: find references for the underlying libraries.

How to cite IAML
================

When reporting a study, cite the software and identify the version or commit
used. See :ref:`citing-iaml` for the suggested citation.


Contribute to IAML
==================

IAML is developed by IIAS. Source code and issue tracking are hosted in the
`IAML GitHub repository <https://github.com/IIAS-Research/iaml>`_.

We welcome contributions, bug reports and examples of research workflows from
clinical researchers and the wider machine learning community.
