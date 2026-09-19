==============
Explainability
==============

The explanations below apply to a trained
:py:class:`~iaml.candidate.Candidate`. Start with :doc:`quick_start`, then keep
the first candidate returned by ``fit``:

.. code-block:: python

    best_candidate = candidates[0]
    print(best_candidate.describe_steps())
    print(best_candidate.describe_metrics())
    print(best_candidate.bibliography())

Feature importance
==================

Compute SHAP explanations using held-out features. The result is an
:py:class:`~iaml.explanation.Explanation` object, from which you can extract
importance values, Markdown tables and plots:

.. code-block:: python

    explanation = best_candidate.explain_feature_importance(X_test)
    importance = explanation.features_importance()
    table = explanation.to_markdown_shap()
    plots = explanation.to_plots()

This computation can be expensive; use a representative subset of ``X_test``
when exploring a large dataset.

Model performance
=================

Evaluate on held-out features and targets, and request the plots suitable for
the task:

.. code-block:: python

    scores = best_candidate.evaluate(X_test, y_test)
    performance_plots = best_candidate.explain_model_performance(X_test, y_test)

Each plot exposes an ``image`` attribute. In a notebook, display the images
with ``IPython.display.Image``:

.. code-block:: python

    from IPython.display import Image, display

    for plot in performance_plots:
        display(Image(plot.image))
