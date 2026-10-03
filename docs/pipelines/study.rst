:orphan:

.. _pipelines-study:

Configure a complete study
==========================

Configure the calculations around a pipeline: describe a cohort, choose the
validation objective and explain a selected model. Analytical collections use
``use`` and the same editing operations as :doc:`pipeline recipes <guide>`,
while search settings belong to the ``IAML`` constructor.

.. _pipelines-study-phases:

Four parts of a study
---------------------

.. list-table:: Study configuration
   :header-rows: 1
   :widths: 15 25 30 30

   * - Attribute
     - Calculation stage
     - Input
     - Result
   * - ``pipeline``
     - Candidate generation, validation and final training.
     - Search data and data selected for final training.
     - Evaluated candidates and fitted models.
   * - ``metrics``
     - During validation; on test data when ``evaluate`` is called.
     - Targets and the prediction types required by each metric.
     - Scores; the main metric determines ranking.
   * - ``statistics``
     - On demand, before or after ``fit``.
     - Explicitly supplied data, or the last raw search dataset.
     - A descriptive table, without transforming observations.
   * - ``explanations``
     - On demand, after a candidate is fitted.
     - The fitted model, supplied observations and, when required, their targets.
     - Explanation objects and performance plots.

Statistics, metrics and explanations are collections of calculations, rather
than steps inserted before or after the predictor. They do not multiply the
candidate pipelines. Configuring descriptive statistics or explanations does
not execute them at every optimizer iteration.

.. _pipelines-study-collections:

Build analytical collections
-----------------------------

``metrics(...)``, ``statistics(...)`` and ``explanations(...)`` accept component
classes or ``use(...)`` recipes. With arguments, they declare exactly the
requested calculations. Without arguments, they use their default family,
which you can customize by addition or subtraction. A collection runs all its
applicable methods, with no ``choice`` or ``start`` policy.

Analytical parameters configure fixed calculations: ``pos_label`` identifies a
class, ``k`` sets a category count and ``nsamples`` sets SHAP effort. Use plain
values. ``Const`` is accepted but redundant; ``Int`` and ``Float`` domains are
rejected because the optimizer does not tune these definitions.

Aliases identify configured variants and their result keys. They are unique
within each collection; a metric and an explanation may share an alias.
Use ``search.metrics["alias"]`` or the corresponding collection to navigate.
The :ref:`editing contracts <pipelines-ref-editing>` cover ``add``, ``remove``,
``replace``, ``clone`` and collective ``find_all(Class).configure(...)``.

Omitting configuration retains the defaults. An explicit empty list, such as
``IAML(statistics=[], explanations=[])``, disables collective calculations.
A training search still needs a metric to rank candidates.

.. _pipelines-study-example:

A binary classification study
-----------------------------

This example keeps the built-in pipeline and configures its analyses and
search. Pass a recipe to ``build_search(pipeline=...)`` to use your own pipeline.
It expects feature ``DataFrame`` objects and aligned target ``Series`` objects,
with binary labels where ``1`` is the event of interest. Test data stays outside
the search.

.. code-block:: python

   from functools import partial

   from iaml import (
       AccuracyMetric,
       ConfusionMatrixPlot,
       IAML,
       MissingRateStatistic,
       RecallMetric,
       SummaryTableStatistic,
       TopKValueCountsStatistic,
   )
   from iaml.explainers import KernelSHAP
   from iaml.flow import explanations, metrics, statistics, use
   from iaml.optimizers import GeneticOptimizer
   from iaml.splitters import kfold_splitter

   def build_search(pipeline=None):
       return IAML(
           pipeline=pipeline,
           metrics=metrics(
               use(RecallMetric, pos_label=1).named("event_recall"),
               use(RecallMetric, pos_label=0).named("other_recall"),
               use(AccuracyMetric).named("accuracy"),
           ),
           main_metric="event_recall",
           statistics=statistics(
               SummaryTableStatistic,
               MissingRateStatistic,
               use(TopKValueCountsStatistic, k=5).named("categories"),
           ),
           explanations=explanations(
               use(ConfusionMatrixPlot).named("confusion"),
               use(KernelSHAP, nsamples=100).named("shap"),
           ),
           optimizer=GeneticOptimizer,
           splitter=partial(kfold_splitter, nb_folds=3),
           max_duration=60,
           max_stage_duration=30,
           max_workers=1,
           keep_training_history=True,
       )

   def run_study(X_train, y_train, X_test, y_test):
       search = build_search()
       descriptive = search.get_descriptive_statistics(X_train, y_train)
       model = search.fit(X_train, y_train)[0]
       return {
           "search": search,
           "model": model,
           "descriptive": descriptive,
           "validation_scores": dict(model.computed_metrics),
           "test_scores": model.evaluate(X_test, y_test),
           "explanations": model.explain(X_test.iloc[:5], y_test.iloc[:5]),
       }

Call ``run_study`` inside your script's ``if __name__ == "__main__":`` guard,
with your training and test data, and save its return value as ``results``.
The functions do not train when imported.
The example catalogue includes a runnable
script with a bundled dataset.

Both recall variants measure the same candidate. ``event_recall`` ranks it;
``other_recall`` and ``accuracy`` provide additional measurements. Only the five
supplied test rows are explained.

.. _pipelines-study-search:

Choose validation and search settings
--------------------------------------

The constructor separates the recipe's allowed components from how IAML
searches and evaluates them:

* ``optimizer`` receives an optimizer class. ``GeneticOptimizer`` can replace
  components within the declared scope; ``RandomOptimizer`` and
  ``BayesianOptimizer`` tune parameter values. A :ref:`partial start
  <pipeline-guide-start>` therefore requires ``GeneticOptimizer`` or another
  optimizer supporting component replacement.
* ``splitter=partial(kfold_splitter, nb_folds=3)`` uses three validation folds.
  The built-in splitter accounts for classification targets and supplied
  groups. Keep held-out test data separate from these internal folds.
* ``max_duration`` sets the search budget in seconds; ``max_stage_duration``
  limits an individual stage. Initialization and final fitting can add time,
  so the budget is not a wall-clock limit for the complete study.
* ``max_workers`` controls worker concurrency. ``keep_training_history=True``
  records validation audit information in ``search.training_history``.

The example chooses bounded search settings for a walkthrough. Configure
sampling, groups, preprocessing and refit policy in
:ref:`the build guide <build-validation>`. Changing the search strategy does
not broaden the components declared by your recipe or unfreeze its constants.

.. _pipelines-study-metrics:

Choose the objective and name scores
------------------------------------

``main_metric="alias"`` selects a definition in the ``metrics`` collection.
Use an alias when declaring several variants of the same class. Alternatively,
a ``Metric`` instance such as ``F1ScoreMetric(pos_label=1)`` retains its exact
configuration. Ambiguous instance-based selection requires an explicit alias.

The objective's calculation and ranking direction, defined by
``greater_is_better``, stay fixed throughout a launch. Unknown or removed
objectives raise an error rather than being silently added back. Automatic
suitability recommendations do not override an explicitly requested objective;
a candidate is rejected if its main score cannot be computed on every fold.

Aliases remain the result keys in ``computed_metrics``, ``evaluate`` and
``describe_metrics``. In the example they are ``event_recall``, ``other_recall``
and ``accuracy``. Unnamed components keep their usual keys when unambiguous;
variants with colliding output keys require explicit aliases.

.. _pipelines-study-validation:

Read validation and test scores
--------------------------------

``model.computed_metrics`` contains the internal validation scores.
``model.describe_metrics()`` presents these scores, including after external
evaluation. ``model.evaluate(X_test, y_test)`` returns fresh scores using the
fitted model's metric definitions, without replacing validation scores or
reranking the earlier search.

A secondary metric has an aggregate score only when every fold supplies a
valid value. With incomplete coverage, ``model.metric_coverage`` retains the
available fold values and no aggregate enters ``computed_metrics``. The
:ref:`analytical reports <pipelines-study-implementation>` distinguish a
missing result from a zero score. See :doc:`../evaluation` for interpretation
and the limits of internal validation.

.. _pipelines-study-statistics:

Describe data before training
-----------------------------

``get_descriptive_statistics(X, y)`` describes the supplied data without
training a model or replacing the dataset remembered from an earlier search.
Both ``X`` and ``y`` are required because some statistics depend on the target.
Methods are filtered for applicability to the supplied column types and target.

Using ``build_search`` above, adjust or replace the attached collection before
requesting another table:

.. code-block:: python

   search = build_search()
   search.statistics["categories"].configure(k=10)
   detailed = search.get_descriptive_statistics(X_train, y_train)

   search.statistics.replace(
       statistics(SummaryTableStatistic, MissingRateStatistic),
   )
   compact = search.get_descriptive_statistics(X_train, y_train)

The no-argument ``get_descriptive_statistics()`` describes the last raw search
dataset, before pipeline transformations. That dataset may be sampled when
``train_on_n_samples`` is used. Before a dataset is remembered, the call returns
an empty table. Pass data explicitly to describe a whole cohort instead.

``visualize_descriptive_statistics()`` creates figures for the remembered
search dataset. Intermediate preprocessing states are not captured
automatically. See :doc:`../scientific` for descriptive reporting.

Cache entries account for data and statistical configuration. Returned tables
are independent copies, including mutable cells. Explicitly named statistics
use their aliases as row names, or ``alias : label`` for multiple rows;
canonical labels remain available to the plotting functions.

.. _pipelines-study-explanations:

Explain a fitted model
----------------------

``model.explain(X, y=None)`` runs applicable methods from its explanations
collection. It returns a dictionary keyed by alias or an unambiguous default
key. Performance plots retain their plot objects; KernelSHAP returns an
``Explanation``. A target is needed for plots such as the confusion matrix;
a call without ``y`` can run SHAP and skip that plot.

If you saved the dictionary returned by ``run_study`` as ``results``, retrieve
its requested explanations without recalculating them:

.. code-block:: python

   model = results["model"]
   explained = results["explanations"]
   figure = explained["confusion"]
   explanation = explained["shap"]

The returned dictionary contains successful calculations. Check
``model.explanation_report`` when an expected key is absent; an inapplicable
or failed method has a distinct status and reason.

KernelSHAP explains the second ``predict_proba`` column when available,
otherwise ``predict``. Inspect ``model.pipeline.classes_[1]`` to identify that
class; a metric's ``pos_label`` does not change the explained output.
``nsamples`` sets effort per prediction, while the number of observations in
``X`` sets the number of rows explained. SHAP is imported only when requested.

Specialized methods such as ``explain_feature_importance`` and
``explain_model_performance`` keep their signatures and return types, including
when collective explanations are disabled. See :doc:`../explainability` for
interpretation and visualization, and the :ref:`API reference
<pipelines-ref-computations>` for adapter constraints.

.. _pipelines-study-snapshots:

Keep launches stable while editing
----------------------------------

At the beginning of ``fit``, IAML copies and resolves its recipe, collections
and objective. Trained candidates retain independent calculation definitions.
Changing ``search.metrics`` or ``search.explanations`` prepares future launches;
it does not change existing models, their ranking or already returned results.

After training, ``search.describe()`` reports the resolved definitions,
evaluated candidates and observed applicability decisions for the completed
launch. Later edits are not presented as if they had participated in it. See
:ref:`inspection and reconstruction <pipelines-ref-inspection>` to save a report
or reconstruct a recipe.

.. _pipelines-study-implementation:

Read analytical reports and partial results
-------------------------------------------

Each collective calculation records ``success``, ``inapplicable`` or ``error``
for its result key. A failed calculation preserves successful results from
other methods; invalid configuration is rejected before calculations begin.

Use ``search.describe()["analysis_reports"]["statistics"]`` for descriptive
reports. Candidates expose ``metric_report`` for validation, ``evaluation_report``
for external evaluation and ``explanation_report`` for explanations. To inspect
incomplete secondary validation coverage:

.. code-block:: python

   coverage = model.metric_coverage["other_recall"]
   print(coverage["available"], coverage["total"])
   print(coverage["values"])
   print(coverage["complete"])

``available`` counts valid fold values, ``total`` counts all folds, ``values``
contains the available scores and ``complete`` indicates full coverage.

Continue with :doc:`reference` for exact contracts, or run
the complete study,
descriptive statistics before training
and metrics and the objective
from the example catalogue.
