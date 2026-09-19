# IAML — Integrated AutoML for Medical Labs

IAML (Integrated AutoML for Medical Labs) is a Python framework developed by IIAS
to make machine learning more accessible to clinical research teams. It brings
preprocessing, model search and evaluation into one workflow for classification,
regression and survival analysis on tabular data.

Researchers can inspect the steps of a selected pipeline, evaluate its predictions
and generate explanations to discuss with clinicians and data scientists. Modular
components let teams adapt the workflow to their study while keeping the methods
available for review.

## Clinical research workflow

- **Build prediction pipelines:** search preprocessing steps, models and their
  parameters through a Python API.
- **Evaluate a study outcome:** choose the metric and validation strategy, then
  assess the selected candidate on held-out data.
- **Inspect and explain:** describe pipeline steps, compute SHAP explanations and
  generate task-specific performance plots.
- **Document an experiment:** collect method references and optionally retain
  cross-validation records with `keep_training_history=True`.

The [research guide](https://iias-research.github.io/iaml/scientific.html) shows how to use these outputs when
reporting a study and recording the settings needed to repeat an experiment.

## Installation

Use Python 3.10 or later and Git. Install IAML from GitHub:

```bash
python -m pip install "git+https://github.com/IIAS-Research/iaml.git"
```

The distribution is named `PyIAML`; the Python import is `iaml`.

## How to run

Save this example as `example.py` and run it with `python example.py`.
It uses a dataset bundled with scikit-learn, so no dataset download is needed.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from iaml import IAML


def main():
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    automl = IAML(max_duration=30, max_workers=1)
    candidates = automl.fit(X_train, y_train, verbose=0)
    best_candidate = candidates[0]

    predictions = best_candidate.predict(X_test)
    scores = best_candidate.evaluate(X_test, y_test)
    print("Predictions:", predictions[:5])
    print("Test metrics:", scores)


if __name__ == "__main__":
    main()
```

`fit` returns a list of fitted `Candidate` objects ordered by performance.
Prediction and evaluation are methods of a candidate. Keep the `__main__` guard
when running scripts because training uses multiprocessing. `max_duration` sets
the search time budget; initialization and final fitting can take additional time.

## Documentation

The [user guides](https://iias-research.github.io/iaml/) cover data preparation, model search, evaluation
and interpretation:

- [Getting started](https://iias-research.github.io/iaml/quick_start.html): prepare data and run an example.
- [Usage](https://iias-research.github.io/iaml/usage.html): configure a search and evaluate predictions.
- [Explainability](https://iias-research.github.io/iaml/explainability.html): interpret model predictions.
- [Research guide](https://iias-research.github.io/iaml/scientific.html): report methods and record experiment settings.
- [Component status](https://iias-research.github.io/iaml/component_status.html): find available and experimental components.

## Credits

- Rudy MERIEUX
- Robin BOURACHOT
- Hugo RUELLET
- Youssouf DAHLOUK
