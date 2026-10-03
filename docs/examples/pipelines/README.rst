:orphan:

.. _pipeline-examples:

===========================
Pipeline and study examples
===========================

Choose one of the thirteen Python examples for a specific task. Run them in an
environment where IAML is installed; see :doc:`../../quick_start` for installation.

Each Python file provides callable functions. Loading a file does not start training
or analytical calculations; pass your data explicitly to its calculation
functions. Some files register a custom component at import time.

For the concepts behind these examples, see the
:doc:`pipeline guide <../../pipelines/guide>`,
:doc:`study analyses <../../pipelines/study>` and
:doc:`API reference <../../pipelines/reference>`.

Run an example
==============

From the repository root, load a numbered file with ``runpy.run_path``. Its
functions are available in the returned dictionary. For example, save and run
this script to train the explicit pipeline on a bundled dataset:

.. code-block:: python

   import runpy

   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from iaml import IAML

   if __name__ == "__main__":
       example = runpy.run_path(
           "docs/examples/pipelines/01_explicit_pipeline.py",
       )
       pipeline = example["build_pipeline"]()

       X, y = load_breast_cancer(return_X_y=True, as_frame=True)
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, stratify=y, random_state=42,
       )
       search = IAML(pipeline=pipeline, max_duration=30, max_workers=1)
       model = search.fit(X_train, y_train)[0]
       print(model.evaluate(X_test, y_test))

Adjust the path if you downloaded an individual example. Keep the
``__main__`` guard in scripts that train models, because training starts worker
processes. The search budget limits search time; final fitting can add time.
Most ``build_*`` functions only return a recipe or study, so you can inspect
and edit it before deciding to train.

Example catalogue
=================

00 — Train with defaults
------------------------

:download:`00_no_configuration.py <00_no_configuration.py>`

``train(X, y)`` returns the study and its fitted candidates. The unconfigured
preset includes its visible ``main`` and ``minimal`` strategies.

01 — Compose an explicit pipeline
---------------------------------

:download:`01_explicit_pipeline.py <01_explicit_pipeline.py>`

``build_pipeline()`` combines cleaning, a normalization choice, optional SMOTE
and a model choice using ``>>``. ``train(X, y)`` trains that recipe. The eight
possible structures remain subject to applicability checks and the search
budget.

02 — Build by subtraction
-------------------------

:download:`02_subtractive_pipeline.py <02_subtractive_pipeline.py>`

``build_pipeline()`` starts from normalization and predictor families, removes
classes, then configures the allowed forests. Exclusions apply to optimizer
replacements as well as initial candidate generation.

03 — Choose the initial exploration
-----------------------------------

:download:`03_initial_exploration.py <03_initial_exploration.py>`

``build_variants()`` returns ``complete``, ``one_initial_model`` and
``initial_subset`` recipes with the same allowed models and different starts.
Partial starts require an optimizer that can replace components, such as
``GeneticOptimizer``. ``start()`` restores a full start.

04 — Edit the AutoML preset
---------------------------

:download:`04_edit_default_pipeline.py <04_edit_default_pipeline.py>`

``build_pipeline()`` navigates the ``main`` branch, removes alternatives, adds a
named forest variant and changes its initial parameter value.
``add_business_feature(pipeline, feature_class)`` inserts a registered component
before cleaning. The ``minimal`` branch remains independently editable.

05 — Select and configure variants
----------------------------------

:download:`05_inspect_and_configure.py <05_inspect_and_configure.py>`

``build_pipeline()``, ``configure_pipeline(pipeline)`` and
``inspect_pipeline(pipeline)`` demonstrate aliases and collective configuration
with ``find_all(Class).configure(...)``. ``build_with_default_domain()`` and
``build_parameter_variants()`` compare initial values, ``Const``, retained
domains and explicit ``Int`` domains.

06 — Reuse fragments independently
----------------------------------

:download:`06_reuse_and_variants.py <06_reuse_and_variants.py>`

``build_variants(feature_class=None)`` derives independent base, compact and
balanced recipes, plus a clinical variant when a custom class is supplied.
``compose_independent_pipelines()`` shows that reusing a fragment does not share
subsequent edits. ``preparation()`` and ``models()`` expose the reusable pieces.

07 — Add a custom component
---------------------------

:download:`07_custom_component.py <07_custom_component.py>`

``BodyMassIndex`` uses the existing ``Actionable`` and ``is_step`` contracts.
``build_pipeline()`` inserts it into the main strategy. Supply numeric
``height_cm`` and ``weight_kg`` columns when training this recipe; its
transformation derives ``bmi``. The custom registry tag avoids automatic
insertion elsewhere in the preset.

08 — Keep strategies coherent
-----------------------------

:download:`08_coherent_strategies.py <08_coherent_strategies.py>`

``build_pipeline()`` chooses between normalization plus logistic regression and
a forest with its own domains. ``customize_pipeline()`` edits only the forest
branch. ``train(X, y)`` trains the declared strategies. Entire sub-pipelines
use full initial exploration; replacing them during a partial search is
unsupported.

09 — Replace, describe and reconstruct
--------------------------------------

:download:`09_replace_and_inspect.py <09_replace_and_inspect.py>`

``build_variant()`` replaces normalization and returns the base recipe, edited
recipe and attached replacement. ``inspect_recipe(pipeline, base)`` returns
``describe``, ``diff`` and ``to_code`` outputs. ``train_and_inspect(X, y)`` adds
the report of the actual search.

10 — Configure a complete study
-------------------------------

:download:`10_full_workflow.py <10_full_workflow.py>`

``build_search()`` configures the pipeline, metrics, statistics and explanations.
``run_study(X_train, y_train, X_test, y_test)`` returns descriptive tables,
validation and test scores, a confusion matrix and KernelSHAP explanations for
five test rows. Use binary labels ``0`` and ``1``; the configured positive class
is ``1``. Explanations run only when explicitly requested.

11 — Describe data before training
----------------------------------

:download:`11_descriptive_without_model.py <11_descriptive_without_model.py>`

``build_search()`` selects descriptive methods. ``describe_dataset(X, y)``
configures the top category counts and returns a study with its descriptive
table; ``replace_description(X, y)`` replaces the collection and returns a new
table. Neither function trains a model. Applicable methods depend on the column
types and target supplied.

12 — Name metrics and select an objective
-----------------------------------------

:download:`12_metrics_and_objective.py <12_metrics_and_objective.py>`

``build_metrics()`` defines distinct recall variants for binary labels ``1`` and
``0``. ``build_search()`` selects ``event_recall`` as the objective and returns
the study and its recall selection. ``without_secondary_recall()`` returns
independent complete and reduced collections.
``run_study(X_train, y_train, X_test, y_test)`` compares validation and test scores
while retaining each metric's alias.
