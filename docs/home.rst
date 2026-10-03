========
Overview
========

**IAML (Integrated AutoML for Medical Labs)** is a Python framework for research
teams working with tabular clinical data. It combines preprocessing, model
search and evaluation for classification, regression and survival analysis.

IAML offers an integrated workflow with ready-to-use defaults and a pipeline
API for teams that want to define their own methods. Both use the same search,
cross-validation and final fitting engine.

Who is it for?
==============

Clinical researchers and data scientists working with Python and pandas.
Researchers define the study population, outcome, features and evaluation
design; IAML searches prediction pipelines within that design. The selected
methods and parameters remain available for review.

An integrated workflow, ready to use
====================================

Start with ``IAML().fit(X, y)`` to search preprocessing methods, predictors and
their parameters using the built-in pipeline. You do not need to construct
a pipeline or configure each step to obtain trained candidates.

Follow :doc:`quick_start` to run a first analysis. The :doc:`worked_example`
connects the same calls to performance figures, explanations and saved outputs.

A pipeline API for teams going further
======================================

Adapt the preset or compose a recipe from reusable components. Choose the
permitted alternatives, edit parameter domains and configure the analyses
around training. Your own components can join these recipes through the
extension interfaces.

discover_pipelines introduces these possibilities visually.
pipelines/index directs you to the construction, study configuration
and extension guides when you are ready to customize a workflow.

What does a run produce?
========================

The search compares **candidates** by cross-validation. Each candidate combines
preprocessing, a predictor and its scores; retained candidates are then refitted
on the chosen training population, using all training rows by default.
``search.chosen_candidate`` is the selected fitted
candidate, including the preprocessing needed for new observations.

The study continues through the following outputs:

.. list-table::
   :header-rows: 1
   :widths: 23 50 27

   * - Stage
     - Output
     - Guide
   * - Describe and train
     - Cohort summaries on request, then fitted candidates and validation scores.
     - :doc:`usage`
   * - Evaluate
     - Held-out scores and task-specific performance figures.
     - :doc:`evaluation`
   * - Explain
     - Selected methods and feature contributions for predictions.
     - :doc:`explainability`
   * - Report
     - Saved outputs, experiment settings and method references.
     - :doc:`scientific`

Descriptive statistics, figures and explanations are calculated when requested.
They do not all run automatically during ``fit``. See :doc:`architecture` for
how candidate generation, evaluation and optimization fit together.

Project and contributions
=========================

IAML is developed by `IIAS <https://www.iias.fr/>`_. Source code and issue tracking
are available in the
`IAML GitHub repository <https://github.com/IIAS-Research/iaml>`_. Contributions,
bug reports and examples of research workflows are welcome.
