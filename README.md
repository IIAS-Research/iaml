# IAML — Integrated AutoML for Medical Labs

IAML (Integrated AutoML for Medical Labs) is a Python framework developed by IIAS
to make machine learning more accessible to clinical research teams. It brings
preprocessing, model search and evaluation into one workflow for classification,
regression and survival analysis on tabular data.

Researchers can inspect the steps of a selected pipeline, evaluate its predictions
and generate explanations to discuss with clinicians and data scientists.
IAML includes a broad set of built-in methods. **Go further with customization.**
Add your team's preprocessing steps, models, metrics, validation splitters
and search optimizers to adapt the workflow to your research domain.
These contributions can be shared and reused across studies.
The [extension guide](https://iias-research.github.io/iaml/adaptability.html) shows how to get started.

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

Use Python 3.10 or later:

```bash
python -m pip install PyIAML
```

The distribution is named `PyIAML`. The Python import is `iaml`.

## How to run

Save this example as `example.py` and run it with `python example.py`.
It uses a dataset bundled with scikit-learn, so no dataset download is needed.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from iaml import IAML

if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=42
    )
    search = IAML(max_duration=30, max_workers=1)
    search.fit(X_train, y_train)
    chosen_model = search.chosen_candidate
    print(chosen_model.evaluate(X_test, y_test))
```

`chosen_model` is the selected model, including its preprocessing.
`evaluate` scores it on the held-out test set. The dataset labels are `0` for
malignant and `1` for benign. Keep the `__main__` guard because training uses
multiprocessing. The example uses one worker and a 30-second search budget.
Final fitting can take additional time.

## Documentation

The [user guides](https://iias-research.github.io/iaml/) cover data preparation, model search, evaluation
and interpretation:

- [Quick Start](https://iias-research.github.io/iaml/quick_start.html): install IAML and run an example.
- [01 / Build](https://iias-research.github.io/iaml/usage.html): prepare data and configure a search.
- [02 / Evaluate](https://iias-research.github.io/iaml/evaluation.html): assess predictions on held-out data.
- [03 / Explain](https://iias-research.github.io/iaml/explainability.html): inspect methods and interpret feature contributions.
- [Study reporting](https://iias-research.github.io/iaml/scientific.html): save outputs and record experiment settings.
- [Extending IAML](https://iias-research.github.io/iaml/adaptability.html): add reusable methods for your team's research.
- [Component availability](https://iias-research.github.io/iaml/component_status.html): explore the main component families and their API documentation.

## Credits

- Rudy MERIEUX
- Robin BOURACHOT
- Hugo RUELLET
- Youssouf DAHLOUK
