============
03 / Explain
============

Inspect the selected pipeline and explore how its features contribute to
predictions. Use :doc:`evaluation` to assess prediction performance first.

The snippets below continue :ref:`build-classification`, using its trained
``model`` and held-out ``X_test``. In the example script, place them inside
``main()``, after ``fit`` and the evaluation.

Inspect the pipeline
====================

Read the descriptions of the selected preprocessing steps and predictor. The
structured summary also includes training-only resampling steps and their
configured parameters:

.. code-block:: python

    for description in model.describe_steps():
        print(description)

    pipeline_summary = model.pipeline_audit_summary()
    print(pipeline_summary)

See :doc:`scientific` to save this summary and review the component references
for a study report.

Explain a class probability
===========================

For this section, use the :ref:`build-probabilities` configuration when creating
``search``, before fitting. With a binary probability model, IAML explains the
second probability column, ``predict_proba(X)[:, 1]``. Check
``model.pipeline.classes_[1]`` to identify the class being explained. In the
synthetic Build example this is label ``1``; see :ref:`evaluate-positive-class`
for defining the event of interest. A metric's ``pos_label`` does not change
which output SHAP explains.

Start with a few held-out rows to inspect individual predictions:

.. code-block:: python

    import numpy as np

    explained_class = model.pipeline.classes_[1]
    print("Explained class:", explained_class)

    X_explain = X_test.iloc[:5]
    np.random.seed(42)
    explanation = model.explain_feature_importance(X_explain, nsamples=20)
    importance = explanation.features_importance()
    print(explanation.to_markdown_shap())

IAML uses SHAP's KernelExplainer. ``nsamples`` controls its sampling effort for
each explained prediction. It does not select the number of patients in
``X_explain`` or the background data. The small value above is a starting point
for exploration. Increase it and check the stability of the explanation before
reporting results.

The background is the reference dataset retained by the pipeline, normally the
initial generation sample from the search. If no reference data is available,
the supplied features are used as the background. Runtime also depends on that
background's size and the cost of model predictions. The first five rows are
only a demonstration. Select a representative subset for a study-level summary.

Interpret the output
====================

``features_importance()`` and the Markdown table report the **mean absolute SHAP
value** for each feature across the explained rows. They summarize contribution
magnitude, not its direction. Signed, per-row contributions remain available in
``explanation.shap_values``.

These contributions describe the fitted model relative to its background data.
They do not establish that changing a clinical feature causes a change in the
outcome.

For a multiclass classifier, the current interface still explains only the
second class. It has no class selector. Without ``predict_proba``, it explains
the output of ``predict`` instead. For regression this is the predicted value.
For survival it is the predictor's output, not a survival probability at a
chosen time. The binary example above should not be interpreted as covering
all these cases.

Display or save a figure
========================

Create a bar plot of the contributions and save its image as a PNG:

.. code-block:: python

    from pathlib import Path

    plot = explanation.to_plot("bar")
    Path("shap_importance.png").write_bytes(plot.image)

In a notebook, display the same image directly:

.. code-block:: python

    from IPython.display import Image, display

    display(Image(data=plot.image))

The same ``plot.image`` interface is available for the performance figures
described in :doc:`evaluation`.

.. _configure-explanations-for-the-study:
.. _explain-configure-collection:

Reuse explanation settings across candidates
============================================

For repeated analyses, an optional ``explanations`` collection records the
methods and parameters to use. Add this configuration to the binary Build
example before ``fit``; fitting itself does not calculate these explanations:

.. code-block:: python

    from iaml import ConfusionMatrixPlot
    from iaml.explainers import KernelSHAP
    from iaml.flow import explanations, use

    search.explanations.replace(explanations(
        use(ConfusionMatrixPlot).named("confusion"),
        use(KernelSHAP, nsamples=20).named("shap"),
    ))

After fitting, request calculations on the held-out observations:

.. code-block:: python

    results = model.explain(X_test.iloc[:5], y_test.iloc[:5])
    print(model.explanation_report)
    if "shap" in results:
        print(results["shap"].to_markdown_shap())
    if "confusion" in results:
        print(results["confusion"].title)

The result dictionary uses the configured aliases. Each successful value
keeps its native type: an ``Explanation`` for SHAP or a plot object for a
performance figure. Calculations run only when requested. The report
distinguishes successful, inapplicable and failed methods; an individual
failure preserves the other successful outputs. SHAP is imported on demand.
Without ``y``, methods that need observed outcomes are inapplicable.

Each candidate retains the configuration from its training run; later edits
to ``search.explanations`` apply to the next run. The direct calls illustrated
above remain available. See :ref:`pipelines-study-explanations` for collection
editing and :ref:`pipelines-study-implementation` for report details.

Next: :doc:`scientific` explains which settings and outputs to retain for a
study report.
