.. _evaluate-model:

=============
02 / Evaluate
=============

These snippets continue the :ref:`build-classification` script with its trained
:py:class:`~iaml.candidate.Candidate`, named ``model``. Place them inside
``main()``, after fitting. Keep ``X_test`` and ``y_test`` separate from the search
so they measure performance on observations the search did not use.

.. _evaluate-cv:

Distinguish search scores from test scores
==========================================

``describe_metrics()`` reports the stored internal cross-validation scores used
during the search. ``evaluate(X_test, y_test)`` computes a new dictionary of
scores using the fitted model. It does not replace those stored CV scores.

.. code-block:: python

    print("Internal cross-validation:")
    print(model.describe_metrics())

    test_scores = model.evaluate(X_test, y_test)
    print("Held-out test set:")
    for name, score in test_scores.items():
        print(f"{name}: {score:.3f}")

The selected metric determines the candidate ranking. Other reported metrics
depend on the task and on what the model can compute. For example, ROC AUC
requires probability predictions. An unavailable test metric is omitted from
the returned dictionary. An absent result should not be interpreted as zero.

To retain per-fold records, create the search with
``keep_training_history=True`` before fitting, as shown in :ref:`build-history`.
Inspect ``search.training_history`` for evaluated candidates and their fold
results. See :doc:`scientific` for reporting an experiment.

.. _evaluate-predictions:

Make predictions
=================

Use the same feature names and definitions as during training, without the
outcome or patient grouping columns:

.. code-block:: python

    predictions = model.predict(X_test)
    print(predictions[:5])

Classification returns predicted labels, regression returns numerical
predictions, and survival returns the chosen estimator's prediction, commonly
a risk score. A survival risk score is not a probability at a specified time.
Its meaning depends on the selected model.

Class probabilities
-------------------

``predict_proba`` is available only when the fitted classifier supports it.
For example, LinearSVC predicts classes but does not provide probabilities.
Calling ``model.predict_proba`` then raises ``AttributeError``.

For the ``0/1`` example in :ref:`build-classification`, check the class
order before naming the probability columns:

.. code-block:: python

    if hasattr(model.pipeline, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        print("Column order:", model.pipeline.classes_)
        print(probabilities[:5])
    else:
        print("This candidate does not provide class probabilities.")

In that synthetic example, label ``1`` has no clinical meaning. Define the
event of interest explicitly when using your own data. With textual labels, some
predictors encode classes internally. Check the fitted predictor's label
encoder before assigning clinical meanings to probability columns.

.. _evaluate-positive-class:

Choose the positive class
=========================

For ``PrecisionMetric``, ``RecallMetric`` and ``F1ScoreMetric``:

- ``0/1``, ``False/True`` and ``-1/1`` use ``1`` as the positive class.
- Other binary labels use the last label in sorted order.
- Candidate evaluation uses the training labels to resolve this convention,
  including when a test subset contains only one class.

Set ``pos_label`` explicitly when the study's event of interest differs from
that convention. For a direct calculation, use labels matching your target:

.. code-block:: python

    from iaml import PrecisionMetric, RecallMetric, F1ScoreMetric

    predictions = model.predict(X_test)
    metrics = [
        PrecisionMetric(pos_label=1),
        RecallMetric(pos_label=1),
        F1ScoreMetric(pos_label=1),
    ]
    for metric in metrics:
        score = metric.compute(y_test, predictions, y_train=y_train)
        print(f"{metric}: {score:.3f}")

These three metrics return zero for undefined binary scores, use weighted
averaging for multiclass targets, and sample averaging for multilabel targets.
Changing ``pos_label`` on one metric does not reconfigure other metrics or
plots. The binary plotting examples below use numerical ``0/1`` labels, with
the event of interest encoded as ``1``.

.. _evaluate-plots:

Inspect performance plots
==========================

The available plot families are:

- **Classification:** confusion matrix, classification report and prediction
  errors. Binary ROC and precision-recall curves additionally require
  probabilities and both classes in the evaluation data.
- **Regression:** residuals and predicted-versus-observed values.
- **Survival:** Kaplan–Meier comparison and time-dependent AUC have additional
  prediction and follow-up requirements, described below.

For a classifier without probabilities, request a plot that uses class labels:

.. code-block:: python

    from pathlib import Path
    from iaml import ConfusionMatrixPlot

    plot = ConfusionMatrixPlot().compute(model.pipeline, X_test, y_test)
    Path("confusion-matrix.png").write_bytes(plot.image)

To request all classification performance plots for the ``0/1`` Build example,
use the :ref:`build-probabilities` configuration when creating ``search``, before
fitting. This selects an objective that requires probability predictions:

.. code-block:: python

    from pathlib import Path

    plots = model.explain_model_performance(X_test, y_test)
    for index, plot in enumerate(plots, start=1):
        Path(f"performance-{index}.png").write_bytes(plot.image)

Open the generated PNG files to inspect the results. In a notebook, display a
plot with ``display(Image(plot.image))`` after importing
``Image`` and ``display`` from ``IPython.display``. These calculations add time
after training.

Survival plotting limits
-------------------------

Survival plots are not interchangeable across models. A Kaplan–Meier comparison
requires predicted survival functions. Time-dependent AUC uses risk predictions
and the training targets to estimate censoring. Supply the relevant
``X_train`` and ``y_train`` when using these plot classes, and check that
evaluation times fall within the supported follow-up range.

The current ``explain_model_performance`` method requests every plot registered
for the task. Its survival set includes a cumulative-hazard plot whose required
prediction method is not exposed by ``IAMLPipeline``. Therefore, do not use that
all-plots call as a general survival example. Select a compatible plot explicitly
and check its API requirements, or start with the numerical survival metrics
returned by ``evaluate``.

Next, use :doc:`explainability` to inspect pipeline steps and feature contributions.
