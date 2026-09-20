========
Overview
========

**IAML (Integrated AutoML for Medical Labs)** is a Python framework for research
teams working with tabular clinical data. It combines preprocessing, model
search and evaluation for classification, regression and survival analysis.

Who is it for?
==============

IAML is intended for clinical researchers and data scientists who work with
Python and pandas. It automates the search for prediction pipelines while
keeping the selected methods and parameters available for review. Researchers
define the study population, outcome, features and evaluation design.

Adapt it to your research
=========================

IAML includes a broad set of preprocessing, modeling and evaluation methods.
Teams can extend this set with steps, predictors, metrics, validation splitters
or search optimizers that reflect the practices and requirements of their
research domain. These contributions can be shared and reused across studies.
The :doc:`adaptability` guide explains how to implement them in your own
Python modules.

What does a run produce?
========================

The search returns trained **candidates**: each contains a preprocessing pipeline,
a predictor and its cross-validation scores. You can then:

- Evaluate predictions on data held out from the search.
- Inspect the selected steps and their configuration.
- Request feature explanations and performance plots supported by the model.
- Export method summaries and optionally retain cross-validation records.

IAML uses genetic search by default. See :doc:`architecture` for how generation,
evaluation and optimization fit together.

Where to start
==============

Explore :doc:`worked_example` to see a pipeline's evaluation, plots and
explanations on a concrete dataset.

Follow the :doc:`quick_start` to obtain a trained pipeline, evaluated on held-out
data and ready to make predictions. The :doc:`usage`, :doc:`evaluation` and
:doc:`explainability` guides explain the choices behind these results and help
you refine the workflow for your study.
The :doc:`scientific` guide explains how to retain outputs and document the
settings used in an analysis, including the software citation.

Project and contributions
=========================

IAML is developed by `IIAS <https://www.iias.fr/>`_. Source code and issue tracking
are available in the
`IAML GitHub repository <https://github.com/IIAS-Research/iaml>`_. Contributions,
bug reports and examples of research workflows are welcome.
