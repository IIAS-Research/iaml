# IAML

IAML (Incremental AutoML) is a Python framework developed by IIAS for building,
optimizing and explaining machine learning pipelines for tabular data.

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
