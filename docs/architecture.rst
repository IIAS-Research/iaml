============
How it works
============

This page explains how IAML turns a pipeline definition into evaluated and
fitted results, and how its core classes divide that work. For training
settings, start with :doc:`usage`; for component implementation, see
:doc:`adaptability`.

Recipes and fitted pipelines
============================

A **recipe** describes permitted pipeline structures and parameter domains;
a **candidate** is one concrete pipeline evaluated during the search.
At the start of ``fit``, IAML copies and resolves the recipe into execution
steps. Each run therefore has its own search scope and analytical definitions.
Editing a recipe later does not change its previously fitted candidates.

The built-in recipe combines a full ``main`` strategy and a ``minimal`` strategy
for predictors needing little preparation. A custom recipe defines its own
branches. Both entry points use the same execution engine described below;
:doc:`pipelines/guide` covers recipe construction and editing.

From data to a fitted pipeline
==============================

1. **Generate candidates.** IAML selects components whose tags and suitability
   checks match the data. It explores combinations of cleaning, feature
   selection, normalization, resampling and prediction steps. The default
   genetic search starts with a limited set of preprocessing choices and
   explores alternatives during optimization.
2. **Evaluate candidates.** Each pipeline is fitted and evaluated with
   cross-validation. Each fold learns preparation and model state from its
   training observations. The main metric determines candidate ranking.
3. **Optimize candidates.** The default genetic optimizer changes pipeline
   components and parameters, retaining promising candidates for further
   evaluation. Search continues until the time budget, patience limit or
   optimizer stopping condition is reached.
4. **Refit the selected pipelines.** IAML fits the best candidates on the
   training data and returns them in ranked order. The first returned candidate
   is also available as ``chosen_candidate``.

.. figure:: architecture_flow_diagram.png
   :alt: IAML training loop showing candidate generation, optimization, cross-validation and stopping conditions
   :align: center
   :width: 100%

   Training loop illustrated with Bayesian optimization. The default search
   uses genetic mutations. Click to enlarge.

Choose the validation strategy and main metric for the study before starting
the search. Cross-validation scores guide model selection. Use a separate
evaluation dataset to assess the selected pipeline. See :doc:`usage` for
validation, sampling, refitting and time-budget settings, and :doc:`evaluation`
for interpreting scores and assessing held-out data.

Components and extensions
=========================

Most studies interact with ``IAML`` and the selected ``Candidate``. The classes
below explain how data, pipeline construction and evaluation fit together.
Follow a class name for its API reference.

The search and its results
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Role
   * - :py:class:`~iaml.iaml.IAML`
     - Coordinates candidate generation, validation, optimization and final
       fitting. Holds the search settings and exposes the selected candidate
       through ``chosen_candidate``.
   * - ``iaml.flow`` recipes
     - Describe the pipeline's structure and search scope, along with metric,
       statistic and explanation collections. Compilation creates fresh
       execution instances for each run.
   * - :py:class:`~iaml.dataset.Dataset`
     - Carries features, targets, optional groups and detected data types.
       Steps use this information to check applicability. Splitting and
       sampling keep features, targets and groups together.
   * - :py:class:`~iaml.candidate.Candidate`
     - Combines a pipeline with its dataset, metrics and validation results.
       Provides prediction, held-out evaluation, pipeline descriptions,
       explanations and method references.
   * - :py:class:`~iaml.iaml_pipeline.IAMLPipeline`
     - Holds the ordered transformations, training-only resamplers and final
       predictor. Reuses fitted transformations when predicting new
       observations, without resampling them.

After fitting, ``search.chosen_candidate`` gives access to the results and
reporting methods. Its ``pipeline`` attribute contains the fitted
``IAMLPipeline``, also exposed as ``search.chosen_model``.

Steps and their organization
----------------------------

:py:class:`~iaml.step.Step` defines the shared contract for suitability,
configuration, tags and method references.
:py:class:`~iaml.actionable.Actionable` is the base for concrete transformations,
resampling operations and predictors.
:py:class:`~iaml.predictor.Predictor` specializes it to wrap a learning model
and expose its prediction methods. Probability and survival outputs depend
on the wrapped estimator's capabilities.

Tags associate registered steps with search stages. Compiled meta steps control
whether a stage executes components in order, selects them adaptively or
branches into alternatives.

Evaluation, optimization and reporting
--------------------------------------

* :py:class:`~iaml.metric.Metric` defines how predictions are scored, which
  prediction method is needed and whether larger or smaller values are
  preferable. The main metric controls candidate ranking.
* :py:class:`~iaml.optimizers.optimizer.Optimizer` proposes the next pool from
  evaluated candidates. Genetic optimization can change steps and parameters.
  Bayesian optimization tunes parameters within an existing pipeline structure.
* :py:class:`~iaml.statistic.Statistic` computes descriptive summaries of the
  dataset, such as missing-value counts or feature distributions.
* :py:class:`~iaml.plot.Plot` provides the common interface for figures and their
  export. Specialized classes cover descriptive plots, model performance and
  SHAP views.

Validation splitters are callables. They receive a ``Dataset`` and yield pairs
of training and validation ``Dataset`` objects, allowing study-specific split
strategies without a new base class.

See :doc:`adaptability` for examples of implementing and connecting these
extensions, :doc:`explainability` and :doc:`scientific` for using their outputs,
and :doc:`component_status` to browse component families and their API.
