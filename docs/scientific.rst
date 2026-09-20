===============
Study reporting
===============

Keep a record of the selected pipeline, its evaluation and the choices made
during the study. :doc:`usage` covers building a search, :doc:`evaluation`
covers held-out evaluation, and :doc:`explainability` covers pipeline inspection
and feature contributions.

The snippets below continue :ref:`build-classification` with ``search`` and its
selected ``model``. In the example script, place them inside ``main()``, after
``fit`` and evaluation.

Save the pipeline summary and search history
============================================

To retain cross-validation records, set ``keep_training_history=True`` when
constructing ``IAML``, before fitting, as shown in :ref:`build-history`.
After ``fit``, ``search.training_history`` contains the collected pipeline
configurations, fold-level results and evaluation statuses. These records stay
in memory and are reset by the next ``fit``.

Export the selected pipeline's configuration and the collected history before
starting another search:

.. code-block:: python

    import json
    from pathlib import Path

    audit = {
        "pipeline": model.pipeline_audit_summary(),
        "training_history": search.training_history,
    }
    Path("iaml_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

The audit uses simple, serializable values. Some configuration objects are
represented as strings. The strict JSON export rejects non-finite numbers such
as NaN or infinity if they occur in a metric, so inspect those results before
exporting them.

This file records configurations and cross-validation results. It does not
contain the fitted model, the observations or the row indices of each fold.
Final fitting is not part of the history, and evaluations interrupted before
returning a result may be absent. Preserve the study's data-selection and split
records separately.

Report results with their context
=================================

Keep the held-out scores returned by ``model.evaluate(X_test, y_test)`` separate
from the search's cross-validation scores. ``model.describe_metrics()`` describes
the latter, including after a call to ``evaluate``. Use the workflow in
:doc:`evaluation` to compute and interpret the held-out results, and
:doc:`explainability` to save figures as image files.

Record the following alongside the outputs:

- The cohort definition, inclusion criteria, features, outcome definition and,
  for classification, the label representing the event of interest.
- The training and evaluation populations, split method, any patient or site
  grouping, and the retained row selections for each split.
- The objective metric and its parameters, search budget, optimizer, sampling
  settings and final fitting population. Record ``train_on_n_samples`` and
  ``refit_on_sample`` when used. The search and final fit can use different rows.
- Random seeds, IAML version or commit, and dependency versions. For SHAP,
  record the explained output, rows, background and sampling effort as well.

The audit is one part of this record. It does not capture all these settings
automatically or guarantee identical results on another run.

.. _method-bibliography:

Review method references
========================

Components declare references that can help prepare the methods section:

.. code-block:: python

    print(model.bibliography())

Review this list against the actual pipeline before citing it. The current
bibliography omits training-only resampling steps, so methods such as SMOTE may
need additional references. It also includes a SHAP reference whether or not
SHAP explanations were computed. The pipeline summary helps identify the
methods actually used.

.. _citing-iaml:

Citing IAML
===========

If you use IAML in a study, cite the software release used for the analysis.
For version 1.0.0, use:

    Merieux, R., Ruellet, H., Bourachot, R., Dahlouk, Y., Blanchard, F.,
    & Vuiblet, V. (2026).
    *IAML: Integrated AutoML for Medical Labs* (Version 1.0.0) [Computer software].
    https://github.com/IIAS-Research/iaml/releases/tag/v1.0.0

The same reference in BibTeX:

.. code-block:: bibtex

    @misc{iaml_1_0_0,
      author       = {Merieux, Rudy and Ruellet, Hugo and Bourachot, Robin
                      and Dahlouk, Youssouf and Blanchard, Fr{\'e}d{\'e}ric
                      and Vuiblet, Vincent},
      title        = {{IAML}: Integrated {AutoML} for Medical Labs},
      year         = {2026},
      howpublished = {Computer software},
      note         = {Version 1.0.0},
      url          = {https://github.com/IIAS-Research/iaml/releases/tag/v1.0.0}
    }

Use ``python -m pip show PyIAML`` to check the installed version. For another
release, update the version, year and release URL in the reference. For a
development checkout, cite the full commit identifier and its GitHub URL.

Cite the methods used in your pipeline as well, using the bibliography described
above to identify the relevant references.
