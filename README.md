# IAML — Integrated AutoML for Medical Labs

IAML (Integrated AutoML for Medical Labs) is a Python framework developed by IIAS
for clinical research teams working with tabular data. It supports classification,
regression and survival analysis through two complementary strengths:

- **An integrated workflow with very little configuration.** Describe your data,
  train prediction pipelines with AutoML, evaluate the selected model and request
  explanations and study outputs. Built-in components and an automatic pipeline
  let you start with a few lines of Python.
- **A pipeline API for teams going further.** Compose complete pipelines with
  `>>`, choose or exclude methods, edit reusable fragments and configure parameter
  domains. The same API configures metrics, descriptive statistics and explanations,
  and lets your team's components become reusable building blocks.

Both paths use IAML's training, cross-validation and model selection workflow.
Start with the defaults or build a recipe around your study's methods.

## Installation

Use Python 3.10 or later:

```bash
python -m pip install PyIAML
```

The distribution is named `PyIAML`. The Python import is `iaml`.

## Use the integrated workflow

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
    print(search.get_descriptive_statistics(X_train, y_train))
    search.fit(X_train, y_train)
    chosen_model = search.chosen_candidate
    print(chosen_model.evaluate(X_test, y_test))
```

`chosen_model` is the selected model, including its preprocessing.
`evaluate` scores it on the held-out test set. The dataset labels are `0` for
malignant and `1` for benign. Keep the `__main__` guard because training uses
multiprocessing. The example uses one worker and a 30-second search budget.
Final fitting can take additional time.

Request descriptive statistics, evaluation, plots and explanations when you
need them. The [worked example](https://iias-research.github.io/iaml/worked_example.html)
shows the model, performance plots and SHAP outputs produced by this workflow.

## Compose and adapt complete pipelines

Pipeline configuration is optional. To see what it enables, explore
[Customize IAML](https://iias-research.github.io/iaml/pipelines/guide.html),
then try the recipe below when your study needs specific methods.

Build a recipe from reusable components, then train it through the same `fit`
and `evaluate` calls. For example, impute missing values, choose from the
normalization family except `UnitNormScaler`, and compare two predictors:

```python
from iaml import IAML
from iaml.flow import Int, choice, normalizers, use
from iaml.steps import (
    LogisticRegression, UnitNormScaler, RandomForestClassifier, SimpleImputer,
)

pipeline = (
    use(SimpleImputer).named("cleaning")
    >> normalizers().remove(UnitNormScaler).named("normalize")
    >> choice(
        use(LogisticRegression).named("logistic"),
        use(
            RandomForestClassifier,
            n_estimators=Int(100, 300, initial=150),
        ).named("forest"),
    ).named("predictor")
)
search = IAML(pipeline=pipeline, max_duration=30, max_workers=1)
```

Recipes remain editable after construction: navigate by alias, call `add`,
`remove` or `replace`, and configure all occurrences of a component with
`find_all(Class).configure(...)`. Clone a fragment to reuse it independently.
You can also start from `IAML().pipeline` and adapt its visible `main` and
`minimal` branches.

The [pipeline guide](https://iias-research.github.io/iaml/pipelines/guide.html)
covers construction, editing, fixed values and parameter domains. Continue with
[study configuration](https://iias-research.github.io/iaml/pipelines/study.html)
for metrics, descriptive statistics and explanations. The
[example catalogue](docs/examples/pipelines/README.rst) provides runnable Python examples.

Add custom preprocessing steps, models, metrics, validation splitters or search
optimizers when your domain needs them. These components can be shared across
studies; the [extension guide](https://iias-research.github.io/iaml/adaptability.html)
explains how to implement them.

## Documentation

Start with [Discover IAML](https://iias-research.github.io/iaml/) for a visual
study walkthrough or [Quick Start](https://iias-research.github.io/iaml/quick_start.html)
to run it. The integrated guides follow the study through
[training](https://iias-research.github.io/iaml/usage.html),
[evaluation](https://iias-research.github.io/iaml/evaluation.html),
[explanations](https://iias-research.github.io/iaml/explainability.html) and
[reporting](https://iias-research.github.io/iaml/scientific.html).

The [advanced guide map](https://iias-research.github.io/iaml/pipelines/guide.html)
connects pipeline construction, study configuration and component extensions.
Consult [component availability](https://iias-research.github.io/iaml/component_status.html)
for supported methods and optional dependencies.

## Credits

- Rudy MERIEUX
- Robin BOURACHOT
- Hugo RUELLET
- Youssouf DAHLOUK
