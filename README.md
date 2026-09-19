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

The [research guide](docs/scientific.rst) shows how to use these outputs when
reporting a study and recording the settings needed to repeat an experiment.

## Installation

Use Python 3.10 or later. From a local checkout of this repository, install the
project and its development tools with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
```

For runtime dependencies only, use `uv sync --locked --no-dev`.
To install the library from a local checkout with pip, use `python -m pip install .`.
The distribution is named `PyIAML`; the Python import is `iaml`.

## How to run

Save this example as `example.py` and run it with `uv run --locked python example.py`.
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

## Development checks

Install development dependencies with `uv sync --locked`, then run all tests:

```bash
uv run --locked python -m tests
```

Or run an individual suite:

```bash
uv run --locked python -m tests unit
uv run --locked python -m tests steps
uv run --locked python -m tests statistics
uv run --locked python -m tests integration
```

Run the linter:

```bash
uv run --locked pylint --rcfile=.pylintrc src/iaml
```

CI runs `unit`, `steps` and `statistics` on every branch, plus `integration` on
`main`. Pylint currently reports existing issues without blocking the pipeline.

## Documentation

The guides in [`docs/`](docs/index.rst) and the API reference are built with Sphinx
and AutoAPI:

```bash
uv run --locked sphinx-build -W --keep-going -b html docs public
```

Open `public/index.html` in a browser. CI builds the documentation on every branch
and keeps the HTML as an artifact; deployment runs only on `main`.

The [component status guide](docs/component_status.rst) lists available and
experimental components.

## Repository structure

- `src/iaml/`: library, pipeline components, metrics, statistics and plots.
- `tests/`: unit, component (`steps`), statistics and integration tests.
- `docs/`: Sphinx configuration and reStructuredText documentation.
- `local/`: ignored local experiments and data.

## Credits

- Rudy MERIEUX
- Robin BOURACHOT
- Hugo RUELLET
- Youssouf DAHLOUK
