===========
Quick Start
===========

Installing IAML
===============

Use Python 3.10 or newer. From the repository root, install IAML in your
Python environment:

.. code-block:: bash

    python -m pip install .

For development, ``uv sync --locked`` installs the locked dependencies,
including the testing and documentation tools.

Word2Vec text support
---------------------

Importing IAML and creating a search require no NLTK corpus. To use
``ActWord2Vec`` on text columns, install the English stopwords corpus explicitly:

.. code-block:: bash

    python -m nltk.downloader stopwords

For offline use, install the corpus beforehand and set ``NLTK_DATA`` to its
data directory. IAML never downloads corpora automatically. Word2Vec loads
stopwords only when fitting text and uses word tokenization without the
``punkt`` or ``punkt_tab`` resources.

Train and evaluate a candidate
==============================

``IAML.fit`` searches for pipelines and returns trained
:py:class:`~iaml.candidate.Candidate` objects, ordered by score. Use the first
candidate to predict and evaluate on data that was held out from the search.
Pass pandas DataFrames for both features and targets.

Save this example as ``example.py`` and run ``python example.py`` (or
``uv run --locked python example.py`` in the development environment):

.. code-block:: python

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    from iaml import IAML


    def main():
        data = load_breast_cancer(as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(
            data.data,
            data.target.to_frame(),
            test_size=0.2,
            random_state=42,
            stratify=data.target,
        )

        search = IAML(max_duration=30, max_workers=1)
        candidates = search.fit(X_train, y_train)
        best_candidate = candidates[0]

        predictions = best_candidate.predict(X_test)
        print(predictions[:5])
        print(best_candidate.evaluate(X_test, y_test))


    if __name__ == "__main__":
        main()

The main guard is needed when worker processes start a new Python interpreter.
``max_duration`` sets the search budget in seconds; initialization and fitting
already running candidates can extend the total runtime.

Basic customization
===================

The default metric depends on the target: balanced accuracy for classification,
R² for regression, and IPCW concordance for survival. To choose a metric
explicitly, pass a metric instance:

.. code-block:: python

    from iaml import IAML, RocAucMetric

    search = IAML(
        main_metric=RocAucMetric(),
        max_workers=4,
        max_duration=300,
    )

Call ``search.fit`` in the same way as in the complete example. ROC AUC is
intended for binary classification. See :doc:`usage` and the API reference for
the other settings.

Explain a trained candidate
===========================

After training, use ``best_candidate`` from the example above:

.. code-block:: python

    print(best_candidate.describe_steps())
    print(best_candidate.describe_metrics())

    explanation = best_candidate.explain_feature_importance(X_test)
    print(explanation.to_markdown_shap())
    print(explanation.to_markdown_plots())

    plots = best_candidate.explain_model_performance(X_test, y_test)
    print(best_candidate.bibliography())

SHAP explanations and performance plots are computed on demand and require
additional time.

Next steps
==========

- :doc:`usage`: training parameters, metrics and predictions.
- :doc:`architecture`: how IAML constructs and evaluates pipelines.
- :doc:`explainability`: inspect a trained pipeline and its predictions.
- :doc:`adaptability`: add components or customize the search.
- :doc:`scientific`: reporting models and their references.
