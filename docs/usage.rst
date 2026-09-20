.. _build-model:

==========
01 / Build
==========

This chapter and the following guides help you **understand and tailor your
workflow**: prepare study data, choose validation settings, interpret results
and document the methods used. For a first pipeline with minimal setup, start
with the :doc:`quick_start`.

Here, choose an outcome, prepare your study table and search for a prediction
pipeline. IAML compares candidates by cross-validation, then fits the selected
pipeline on the training data.

.. _build-data:

Prepare features and an outcome
===============================

``X`` is a pandas DataFrame: one observation per row and one feature per column.
Keep the outcome out of ``X``. ``y`` must contain one target per row in the same
order. Matching index labels do not replace this positional alignment.

- **Classification:** a Series or single-column DataFrame of class labels.
- **Regression:** a Series or single-column DataFrame of numerical outcomes.
- **Survival:** a Series of ``(event, time)`` tuples, where ``event=True`` means
  that the event occurred and ``False`` means censored. Use positive follow-up
  times in a consistent unit. A DataFrame with separate event and time columns
  is not accepted directly by ``fit``.

For your own CSV file, separate features and target before splitting the data:

.. code-block:: python

    import pandas as pd

    study = pd.read_csv("study.csv")
    X = study.drop(columns=["outcome", "patient_id"])
    y = study["outcome"]

For survival, construct the target explicitly:

.. code-block:: python

    X = study.drop(columns=["event", "follow_up_months", "patient_id"])
    y = pd.Series(
        list(zip(study["event"], study["follow_up_months"])),
        index=study.index,
        name="survival",
    )

Use boolean or ``0/1`` event values, not strings such as ``"yes"`` and ``"no"``.
Keep patient identifiers for grouping rather than using them as predictors.
See :ref:`build-validation`.

.. _build-classification:

Classification: a complete example
==================================

This small synthetic dataset is generated locally. Save the following as
``build_example.py`` and run ``python build_example.py``. It uses one worker and
80 observations, with 20 reserved for evaluation.

.. code-block:: python
    :name: build-classification-example

    import pandas as pd
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    from iaml import IAML


    def make_example_data():
        features, target = make_classification(
            n_samples=80, n_features=4, n_informative=3,
            n_redundant=0, random_state=42,
        )
        X = pd.DataFrame(features, columns=["x1", "x2", "x3", "x4"])
        y = pd.Series(target, name="outcome")
        return train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=42,
        )


    def main():
        X_train, X_test, y_train, y_test = make_example_data()
        search = IAML(max_duration=30, max_workers=1)
        search.fit(X_train, y_train)
        model = search.chosen_candidate
        print(model.describe_steps())
        print(model.evaluate(X_test, y_test))


    if __name__ == "__main__":
        main()

Keep the main guard: training starts worker processes. After fitting,
``search.chosen_candidate`` gives the selected, fitted
:py:class:`~iaml.candidate.Candidate`. ``fit`` also returns a list of fitted
candidates, ordered by the selected metric. By default, it returns one
candidate. Request more with
``search.fit(X_train, y_train, n_candidates=3)``.

The following guides reuse ``search``, ``model`` and the training/test variables
from this script. Place their follow-up snippets inside ``main()``, after
fitting. The labels here are synthetic ``0/1`` outcomes with no clinical meaning.

.. _build-regression:

Regression: a continuous outcome
================================

Replace only ``make_example_data`` in the complete script above with this
function. Keep its imports, ``main`` and main guard. The target is continuous,
so the default objective becomes R².

.. code-block:: python
    :name: build-regression-example

    def make_example_data():
        from sklearn.datasets import make_regression

        features, target = make_regression(
            n_samples=80, n_features=4, noise=10, random_state=42,
        )
        X = pd.DataFrame(features, columns=["x1", "x2", "x3", "x4"])
        y = pd.Series(target, name="measurement")
        return train_test_split(X, y, test_size=0.25, random_state=42)

IAML infers the task from the target values. Check the inferred task when your
outcome has only a few distinct numerical values, such as an ordinal score.

.. _build-survival:

Survival: an event and a follow-up time
=======================================

Alternatively, replace ``make_example_data`` with this function. The 80
synthetic observations contain both events and censored follow-up times.
The default objective becomes IPCW concordance.

.. code-block:: python
    :name: build-survival-example

    def make_example_data():
        import numpy as np

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "age": rng.integers(35, 85, size=80),
            "marker": rng.normal(size=80),
        })
        event = rng.random(80) < 0.7
        time = rng.uniform(1, 24, size=80)
        y = pd.Series(list(zip(event, time)), name="survival")
        return train_test_split(X, y, test_size=0.25, random_state=42)

These examples demonstrate the input format and workflow. Their synthetic
scores are not evidence of clinical performance.

.. _build-validation:

Separate test data and cross-validation
=======================================

Reserve a test set before the search. IAML's default internal validation uses
five folds: stratified folds for classification, ordinary folds for regression
and survival. Preprocessing is fitted within each training fold.

For repeated observations from the same patient, split the external test set
by patient as well. Passing groups to IAML only controls its internal folds.
Use enough distinct groups, and enough observations per class, for the chosen
number of folds.

If ``X_train`` contains a grouping column, pass its name. IAML removes it from
the predictors:

.. code-block:: python

    search.fit(X_train, y_train, groups_columns=["patient_id"])
    model = search.chosen_candidate

Alternatively, pass a separate grouping DataFrame whose columns are absent
from the features:

.. code-block:: python

    patient_groups = X_train[["patient_id"]]
    search.fit(
        X_train.drop(columns="patient_id"), y_train, groups=patient_groups,
    )
    model = search.chosen_candidate

Choose one of these approaches, and omit the identifier from ``X_test`` when
predicting. The default splitter uses stratified group folds for classification
and group folds for the other tasks.

To change the number of internal folds:

.. code-block:: python

    from functools import partial
    from iaml.splitters import kfold_splitter

    search = IAML(
        splitter=partial(kfold_splitter, nb_folds=3),
        max_duration=30,
        max_workers=1,
    )

.. _build-metrics:

Choose the search objective
===========================

The defaults are balanced accuracy for classification, R² for regression and
IPCW concordance for survival. Pass a metric instance to choose another
objective. Its configuration is retained:

.. code-block:: python

    from iaml import PrecisionMetric

    search = IAML(
        main_metric=PrecisionMetric(pos_label=1),
        max_duration=30,
        max_workers=1,
    )

Use a metric appropriate to the task. For example, ``RocAucMetric()`` requires
binary classification and probability predictions, while
``MeanSquaredErrorMetric()`` is a regression objective. IAML handles the score
direction: it maximizes accuracy-type scores and minimizes error metrics.
See :ref:`evaluate-positive-class` before interpreting binary metrics.

.. _build-probabilities:

When you need class probabilities
---------------------------------

For the binary probability examples in :doc:`evaluation` and
:doc:`explainability`, replace the construction of ``search`` in the
classification script with the following, **before calling** ``fit``:

.. code-block:: python

    from iaml import RocAucMetric

    search = IAML(main_metric=RocAucMetric(), max_duration=30, max_workers=1)

ROC AUC requires probability predictions, so the selected candidate must
support them. Use this setting for binary classification, not for the regression
or survival examples above.

.. _build-history:

When you need a search record
-----------------------------

Add ``keep_training_history=True`` when constructing ``IAML``, before fitting,
to retain cross-validation records in ``search.training_history``. This option
can be combined with the metric settings above. See :doc:`scientific` to export
the records. They are not saved to disk automatically.

.. _build-budget:

Control time and dataset size
=============================

- ``max_workers=1`` limits concurrent candidate evaluations. The default is the
  number of available CPU cores.
- ``max_duration`` is a search budget in seconds, not a deadline for the whole
  script. Initialization and final fitting can add time. Plotting is separate.
  The default, ``-1``, sets no global time limit.
- ``max_stage_duration`` caps an evaluation stage. Its default is
  ``max(max_duration / 5, 900)``. A remaining global budget can shorten a stage.
- ``patience``, passed to ``fit``, stops optimization after that many generations
  without improvement. Its default is unlimited when a time budget is set,
  and 20 generations without improvement when ``max_duration=-1``.

For larger studies, limit the initial number of training rows:

.. code-block:: python

    search = IAML(
        train_on_n_samples=1000,
        refit_on_sample=True,
        max_duration=60,
        max_workers=1,
    )

``refit_on_sample=True`` is the default: the selected pipeline is fitted on
the same initial sample. Use ``False`` to search on the sample and then fit on
all training rows. Without a positive sample limit, final fitting uses all
training rows. Further automatic downsizing during the search does not change
the initial sample retained for final fitting. A row limit is not a timeout.

.. _build-text:

Optional Word2Vec text support
==============================

Numerical datasets and importing IAML require no NLTK corpus. To use
``ActWord2Vec`` on text, install the English stopwords corpus beforehand:

.. code-block:: bash

    python -m nltk.downloader stopwords

For offline use, set ``NLTK_DATA`` to the directory containing the installed
corpus. IAML does not download corpora automatically. This component uses
English stopwords and stemming. Its tokenization does not need ``punkt`` or
``punkt_tab``.

Next, use :doc:`evaluation` to assess the selected candidate on held-out data.
