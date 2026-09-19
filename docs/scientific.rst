============================
IAML for Clinical Research
============================

IAML (Integrated AutoML for Medical Labs) helps research teams build and examine
prediction pipelines for tabular clinical data. Its tools support collaboration
between clinical researchers and data scientists by making pipeline methods,
evaluation results and model explanations available for review.

The examples below use ``model = candidates[0]``, the trained candidate
returned by ``candidates = search.fit(X_train, y_train)`` in :doc:`quick_start`.

Inspect pipeline methods
========================

Review the preprocessing steps and predictor selected for the study. Pipeline
descriptions and the references declared by components can help document the
methods used in an analysis:

.. code-block:: python

    print(model.describe_steps())
    print(model.bibliography())

To inspect step names and parameter values in a structured form, use:

.. code-block:: python

    pipeline_summary = model.pipeline_audit_summary()

Evaluate predictions
====================

Assess the selected candidate on data held out from the search. IAML computes
metrics suitable for the task, such as classification, regression or survival
analysis. Report the evaluation population and split strategy alongside the
scores so that the results can be interpreted in the context of the study.

.. code-block:: python

    results = model.evaluate(X_test, y_test)
    print("Performance metrics:", results)

Explore feature contributions
=============================

SHAP explanations describe how features contribute to model outputs for the
data being examined. They can support discussion of model behavior within the
research team.

.. code-block:: python

    explanation = model.explain_feature_importance(X_test)
    importance = explanation.features_importance()
    print(explanation.to_markdown_shap())

See :doc:`explainability` for available plots and how to display them.

Review performance plots
========================

Request plots appropriate to the prediction task, such as ROC curves, confusion
matrices or regression residual plots. The plots can help researchers examine
prediction errors alongside the numerical metrics.

.. code-block:: python

    plots = model.explain_model_performance(X_test, y_test)

    # Display the plots in a Jupyter notebook.
    from IPython.display import Image, display

    for plot in plots:
        display(Image(plot.image))

Record experiment settings
==========================

Set ``keep_training_history=True`` when constructing ``IAML`` to retain
cross-validation records during the search. After ``fit``, inspect
``search.training_history`` for pipeline configurations, fold-level results and
evaluation status. These records are held in memory and reset on the next call
to ``fit``.

To support reproducibility, record the cohort definition, features and outcome,
training and evaluation splits, metric configuration, search budget, random
seeds and software versions alongside the results. The :doc:`usage` guide
explains the training and evaluation settings; :doc:`adaptability` describes
how to add study-specific components.

.. _citing-iaml:

Citing IAML
===========

Cite the software repository when using IAML in a study, and include the version
or commit used for the analysis:

.. code-block:: text

    IAML contributors. IAML: Integrated AutoML for Medical Labs. Software.
    https://github.com/IIAS-Research/iaml

The component references returned by ``model.bibliography()`` can also help
identify methods to cite in the study's methods section.
