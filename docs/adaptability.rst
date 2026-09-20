==============
Extending IAML
==============

IAML is modular throughout. Data preparation, models, validation, search and
reporting are separate building blocks. You can add a block or replace an
existing one to incorporate your team's methods, provided it follows the
corresponding interface. Extensions can live in your study project and be
reused across studies without changing IAML itself.

The main building blocks
========================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Block
     - Role
   * - :ref:`Transformations <extend-transformations>`
     - Prepare or derive features while preserving observations.
   * - :ref:`Resamplers <extend-resamplers>`
     - Change the rows used for training, for example to balance classes.
   * - :ref:`Predictors <extend-predictors>`
     - Learn from the prepared data and predict outcomes.
   * - :ref:`Pipeline organization <extend-composition>`
     - Define the order of steps and which alternatives to explore.
   * - :ref:`Metrics <extend-metrics>`
     - Measure prediction performance and define the search objective.
   * - :ref:`Validation splitters <extend-splitters>`
     - Separate training and validation observations according to the study.
   * - :ref:`Optimizers <extend-optimizers>`
     - Propose new pipeline configurations from evaluated candidates.
   * - :ref:`Descriptive statistics <extend-statistics>`
     - Summarize the dataset independently of model performance.
   * - :ref:`Plots <extend-plots>`
     - Present data, performance or explanations as figures.

See :doc:`component_status` for existing implementations and :doc:`architecture`
for the relationships between the core classes.

How to use these examples
=========================

Save the definitions below in an importable module named ``study_components.py``.
Keep classes and functions at module level so worker processes can import them.
Import the module before constructing ``IAML``. Each example explains how it
joins the workflow, and the final script connects the regression components.

For steps, ``@is_step(...)`` registers the class under tags used by the search.
Define an explicit constructor with no required arguments. The decorator
initializes the base classes, so the constructor only sets the component's
own configuration and state. Keep constructors lightweight and learn from data
in ``fit``, which must reset learned state on each call.

Use descriptive names and unique metric or statistic identifiers. Keep
configurable choices in ``configuration``, give defaults within the allowed
ranges, and change values through ``configure``. Add descriptions and method
references through ``_description`` and ``refs`` when sharing a component.

.. _extend-transformations:

Add a transformation
====================

An :py:class:`~iaml.actionable.Actionable` can transform features. This example
adds body mass index from height in centimetres and weight in kilograms.
Missing or nonpositive measurements produce a missing BMI.

.. code-block:: python
   :name: extension-step

   import pandas as pd
   from iaml import Actionable
   from iaml.decorators.is_step import is_step


   @is_step("features_precleaning")
   class StudyBMI(Actionable):
       name = "Study BMI"
       _description = "Add BMI from height in cm and weight in kg."

       def __init__(self):
           self.optimizable = False

       def suitable(self, dataset):
           return "bmi" not in dataset.X and all(
               column in dataset.X
               and pd.api.types.is_numeric_dtype(dataset.X[column])
               for column in ["height_cm", "weight_kg"]
           )

       def fit(self, dataset):
           return self

       def transform(self, X):
           result = X.copy()
           height = result["height_cm"]
           weight = result["weight_kg"]
           metres = height.where(height > 0) / 100
           result["bmi"] = weight.where(weight > 0) / metres**2
           return result

**Connect it.** Importing the module registers this transformation with the
regular ``features_precleaning`` stage. It can also be placed explicitly in a
workflow, as shown below.

**Design constraints.**

* ``suitable(dataset)`` checks the required columns and types without changing
  the data.
* ``fit(dataset)`` returns ``self``. Learned transformations estimate their
  state from the supplied training data and reuse it in ``transform``.
* ``transform(X)`` returns a DataFrame with the same rows and index. Copy the
  input before changing it and keep the output columns consistent at prediction.

.. _extend-resamplers:

Add a resampler
===============

A resampler is also an ``Actionable``, but implements ``resample(X, y)`` to
change training rows. This example duplicates minority-class observations
using the existing imbalanced-learn dependency.

.. code-block:: python
   :name: extension-resampler

   from imblearn.over_sampling import RandomOverSampler
   from iaml import Actionable
   from iaml.decorators.is_step import is_step


   @is_step("imbalance")
   class StudyOversampling(Actionable):
       name = "Study oversampling"

       def __init__(self):
           self.sampler = None

       def suitable(self, dataset):
           return dataset.type_of_target in ("binary", "multiclass")

       def fit(self, dataset):
           self.sampler = RandomOverSampler(
               sampling_strategy="minority", random_state=42
           )
           return self

       def resample(self, X, y):
           return self.sampler.fit_resample(X, y)

**Connect it.** The ``imbalance`` tag makes it available to the corresponding
search stage. With the default genetic optimizer, resampling alternatives can
be introduced by mutation. In an explicit classification workflow, place
``StudyOversampling()`` after preprocessing and before the predictor.

**Design constraints.**

* Return aligned features and targets, keeping features as a DataFrame.
* Preserve every supplied column. IAML temporarily includes group identifiers
  during resampling and extracts them afterwards. This duplication-based
  example carries those identifiers with their observations.
* Resampling belongs inside each training fold. Validation and prediction
  observations retain their original rows.

.. _extend-predictors:

Add a predictor
===============

Inherit from :py:class:`~iaml.predictor.Predictor` to adapt a learning model.
This small Ridge wrapper exposes one regularization parameter. IAML already
provides Ridge regression, so the example demonstrates the adapter contract.

.. code-block:: python
   :name: extension-predictor

   import numpy as np
   import pandas as pd
   from sklearn.linear_model import Ridge
   from iaml.predictor import Predictor
   from iaml.decorators.is_step import is_step


   @is_step("predictor", "study_regression")
   class StudyRidge(Predictor):
       name = "Study ridge"

       def __init__(self):
           self.configuration = {
               "alpha": {
                   "description": "L2 regularization strength.",
                   "default": 1.0,
                   "range": [0.1, 10.0],
               }
           }

       def suitable(self, dataset):
           return (
               dataset.type_of_target == "continuous"
               and not dataset.X.empty
               and all(pd.api.types.is_numeric_dtype(dtype)
                       for dtype in dataset.X.dtypes)
               and not dataset.X.isna().any().any()
               and np.isfinite(dataset.X.to_numpy(dtype=float)).all()
           )

       def fit(self, dataset):
           self.model = Ridge(**self.passthrough_parameters())
           self.model.fit(dataset.X, dataset.y)
           return self

**Connect it.** Importing the class makes it available through the ``predictor``
tag. Create an instance and call ``configure({"alpha": 0.1})`` to set a value
in an explicit workflow. The additional ``study_regression`` tag separates
this family from built-in predictors when genetic mutations replace steps.

**Design constraints.**

* ``suitable`` declares the supported task and inputs. This wrapper requires
  finite numerical features and a continuous target.
* Each ``fit`` creates a fresh estimator in ``self.model`` and returns ``self``.
  Use only the supplied training data and preserve feature order.
* Forward exposed parameters with ``passthrough_parameters()``. Mark any
  wrapper-only parameter with ``passthrough=False``.
* The base class delegates prediction to the estimator. Probability or survival
  outputs are available only when that estimator supports them.

.. _extend-composition:

Organize or replace pipeline stages
===================================

Meta steps compose other steps. This reusable workflow derives BMI, imputes
missing values, scales features and explores two Ridge configurations.

.. code-block:: python
   :name: extension-composition

   from iaml import (
       ActSimpleImputer, ActStandardScaler, MetaExplorerStep, MetaOrderedStep,
   )
   from iaml.decorators.is_step import is_step


   @is_step("study_workflow")
   class StudyWorkflow(MetaOrderedStep):
       def __init__(self):
           predictors = MetaExplorerStep()
           for alpha in (0.1, 10.0):
               predictor = StudyRidge()
               predictor.configure({"alpha": alpha})
               predictors.add_step(predictor)
           self.add_steps([
               StudyBMI(), ActSimpleImputer(), ActStandardScaler(), predictors,
           ])

**Connect it.** After constructing ``search``, assign
``search.first_step = StudyWorkflow()``. To use only this workflow, also set
``search.minimal_predictor_step = None``. Otherwise IAML retains a separate
branch of minimally preprocessed predictors. The final script applies both
settings.

**Design constraints.**

* ``MetaOrderedStep`` preserves the supplied order. ``MetaStep`` instead
  chooses child steps by priority for the current candidate.
* ``MetaExplorerStep`` creates alternative candidates from the same input.
  Every completed branch must end with a suitable predictor.
* ``MetaPartialExplorerStep`` starts with one choice or no transformation,
  leaving alternatives to later mutations. Use it to limit initial branching.
* Tags identify eligible components, not execution order. Adding a new tag
  requires a stage that uses it. Registration alone does not make a step
  mandatory in every candidate.

Ordinary transformations and predictors reuse the inherited ``run`` method.
If a new orchestration strategy needs its own ``run(candidate)``, decorate it
with ``@runner`` from ``iaml.decorators.runner`` and return a ``Candidate`` or
list of candidates. This preserves IAML's suitability checks and step cache.
For a mandatory transformation across the default search branches,
``IAML(initial_preprocessor=...)`` accepts a clonable scikit-learn transformer
that returns a numerical DataFrame with unchanged rows and index.

.. _extend-metrics:

Add a metric
============

A :py:class:`~iaml.metric.Metric` defines the score and its direction. This
example implements mean absolute error to show the contract for a custom loss.

.. code-block:: python
   :name: extension-metric

   from sklearn.metrics import mean_absolute_error
   from iaml import Metric


   class StudyAbsoluteError(Metric):
       greater_is_better = False

       def __str__(self):
           return "study_absolute_error"

       def suitable(self, X, y, type_of_target):
           return type_of_target == "continuous"

       def compute(self, y, y_pred, **kwargs):
           return float(mean_absolute_error(y, y_pred))

**Connect it.** Pass the instance ``main_metric=StudyAbsoluteError()`` to
``IAML``. Imported metric subclasses are also considered for secondary scores
when suitable.

**Design constraints.** Use a unique string identifier and a constructor with
no required arguments. Return a finite scalar and declare the correct
``greater_is_better`` direction. Error values remain positive in reports.
Metrics use ``predict`` by default. Override ``needed_prediction`` when another
prediction method is required and ensure the selected estimators support it.

.. _extend-splitters:

Add a validation splitter
=========================

A splitter is a callable receiving a :py:class:`~iaml.dataset.Dataset`.
This example uses three folds with separate patient or site groups.

.. code-block:: python
   :name: extension-splitter

   from sklearn.model_selection import GroupKFold


   def study_splitter(dataset):
       if not dataset.has_groups:
           raise ValueError("This study requires patient or site groups.")
       folds = GroupKFold(n_splits=3)
       yield from dataset.split(
           folds.split, groups=dataset.groups.iloc[:, 0]
       )

**Connect it.** Pass ``splitter=study_splitter`` and provide ``groups`` to
``fit``. The example requires at least three distinct groups.

**Design constraints.** Yield pairs of training and validation **Dataset
objects**. ``Dataset.split`` adapts index-based splitters and preserves data
and group metadata. Use groups explicitly when separation is required and
check that training and validation groups are disjoint. Define reproducible
seeds for randomized strategies. See :ref:`build-validation` for study-level
validation choices.

.. _extend-optimizers:

Add an optimizer
================

An :py:class:`~iaml.optimizers.optimizer.Optimizer` receives evaluated candidates
and proposes new ones. This example keeps the best candidate and explores a
small grid around the best ``StudyRidge`` pipeline in one optimization round.

.. code-block:: python
   :name: extension-optimizer

   from time import monotonic
   from iaml import Optimizer


   class StudyOptimizer(Optimizer):
       def __init__(self, duration=None):
           super().__init__()
           self.deadline = float("inf") if duration is None else monotonic() + duration
           self.done = False

       @property
       def finished(self):
           return self.done or monotonic() >= self.deadline

       def run(self, candidates):
           self.done = True
           if not candidates:
               return []
           proposals = [max(candidates)]
           ridge_candidates = [
               candidate for candidate in candidates
               if candidate.pipeline.predictor is not None
               and isinstance(candidate.pipeline.predictor[1], StudyRidge)
           ]
           if not ridge_candidates:
               return proposals
           seed = max(ridge_candidates)
           for alpha in (0.1, 1.0, 10.0):
               if monotonic() >= self.deadline:
                   break
               if alpha == seed.pipeline.predictor[1].get_config("alpha"):
                   continue
               proposal = seed.to_output()
               proposal.pipeline.predictor[1].configure({"alpha": alpha})
               proposals.append(proposal)
           return proposals

**Connect it.** Pass the **class**, ``optimizer=StudyOptimizer``. IAML constructs
it with the remaining search ``duration``, or ``None`` for an unlimited search.

**Design constraints.**

* Expose ``finished`` and implement ``run(candidates)``. Return proposals for
  IAML to evaluate, keeping model fitting outside the optimizer.
* Compare candidates directly or use ``get_main_metric_score()`` to respect
  both maximization and minimization objectives. Retain a strong candidate
  while exploring alternatives.
* Copy candidates with ``to_output()`` before editing them. Use ``configure``
  so parameter changes update the cache fingerprints.
* Bound proposal generation and check the remaining duration. Custom ``run``
  code is not interrupted at its deadline. Its stopping condition does not
  limit initial candidate generation or final fitting.

.. _extend-statistics:

Add a descriptive statistic
===========================

A :py:class:`~iaml.statistic.Statistic` summarizes the data. This example measures
the width between the 10th and 90th percentiles of each numerical feature in
a regression dataset.

.. code-block:: python
   :name: extension-statistic

   import pandas as pd
   from iaml import Statistic


   class StudyCentralWidth(Statistic):
       _description = "Width between the 10th and 90th percentiles."

       def __str__(self):
           return "study_central_width"

       def suitable(self, dataset):
           return (
               dataset.type_of_target == "continuous"
               and not dataset.X.select_dtypes(include="number").empty
           )

       def compute(self, dataset, **kwargs):
           values = dataset.X.select_dtypes(include="number")
           width = values.quantile(0.9) - values.quantile(0.1)
           return pd.DataFrame([width], index=[str(self)])

**Connect it.** Import the class before calling
``search.get_descriptive_statistics()``. IAML discovers suitable subclasses
and includes their rows in the returned table. Results are computed on demand
and cached for that fit.

**Design constraints.** Provide a constructor with no required arguments and
implement ``suitable(dataset)``. Return a DataFrame with unique statistic
identifiers as rows and feature names as columns, without changing the dataset.
For statistics grouped by classification label, follow the existing naming
convention ``<feature>_all`` and ``<feature>_<class>`` so descriptive plots can
group the columns.

.. _extend-plots:

Add a plot
==========

Inherit directly from :py:class:`~iaml.metric_plot.MetricPlot` to add a model
performance figure. This example compares observed and predicted values for
single-output regression.

.. code-block:: python
   :name: extension-plot

   import io
   import numpy as np
   import matplotlib.pyplot as plt
   from iaml import MetricPlot


   class StudyObservedPredicted(MetricPlot):
       title = "Observed and predicted outcomes"
       description = "Compare predictions with the observed outcomes."

       @classmethod
       def suitable(cls, type_of_target):
           return type_of_target == "continuous"

       def compute(self, estimator, X, y, X_train=None, y_train=None, **kwargs):
           observed = np.asarray(y).ravel()
           predicted = np.asarray(estimator.predict(X)).ravel()
           limits = [min(observed.min(), predicted.min()),
                     max(observed.max(), predicted.max())]
           figure, axes = plt.subplots()
           axes.scatter(observed, predicted)
           axes.plot(limits, limits, "--", color="gray")
           axes.set(xlabel="Observed", ylabel="Predicted", title=self.title)
           self._binary_image = io.BytesIO()
           figure.savefig(self._binary_image, format="png", bbox_inches="tight")
           plt.close(figure)
           return self

**Connect it.** After import, the figure is included in
``chosen_model.explain_model_performance(X_test, y_test)`` for regression.
To generate only this figure, call its ``compute`` method directly as in the
final example.

**Design constraints.** Inherit directly from ``MetricPlot`` for automatic
discovery and provide a constructor with no required arguments. Declare the
supported target type and accept the training-data keywords shown above.
Use the fitted pipeline for prediction, without fitting on evaluation data.
Return ``self`` after storing PNG bytes in ``self._binary_image`` as a
``BytesIO``, and close the Matplotlib figure. The example expects nonempty,
finite observed and predicted values.

For descriptive figures, inherit directly from
:py:class:`~iaml.plot.StatisticPlot` instead. Its ``compute(dataframe, **kwargs)``
receives the statistics table. Check the rows it needs and return an exported
figure using the same image contract.

Connect and check the components
================================

Save this separate script beside ``study_components.py``. It uses 48 synthetic
observations, three grouped validation folds and one worker. The explicit
workflow limits the search to the small Ridge family. The classification
resampler is not used in this regression example.

.. code-block:: python
   :name: extension-search

   from pathlib import Path
   import numpy as np
   import pandas as pd
   from iaml import IAML
   from study_components import (
       StudyAbsoluteError, StudyOptimizer, StudyWorkflow,
       StudyObservedPredicted, study_splitter,
   )


   def main():
       rng = np.random.default_rng(42)
       X = pd.DataFrame({
           "height_cm": rng.uniform(150, 190, 48),
           "weight_kg": rng.uniform(50, 100, 48),
       })
       y = pd.Series(0.3 * X["weight_kg"] + rng.normal(0, 1, 48))
       groups = pd.DataFrame({"patient": np.repeat(np.arange(12), 4)})
       X_train, X_test = X.iloc[:36], X.iloc[36:]
       y_train, y_test = y.iloc[:36], y.iloc[36:]
       search = IAML(
           main_metric=StudyAbsoluteError(),
           splitter=study_splitter,
           optimizer=StudyOptimizer,
           max_duration=30,
           max_workers=1,
       )
       search.first_step = StudyWorkflow()
       search.minimal_predictor_step = None
       search.fit(X_train, y_train, groups=groups.iloc[:36])
       chosen_model = search.chosen_candidate
       print(chosen_model.evaluate(X_test, y_test))
       statistics = search.get_descriptive_statistics()
       print(statistics.loc["study_central_width"].dropna())
       plot = StudyObservedPredicted().compute(chosen_model.pipeline, X_test, y_test)
       Path("observed-predicted.png").write_bytes(plot.image)


   if __name__ == "__main__":
       main()

Before reusing an extension, check it on a few observations: expected outputs,
row and group alignment, refitting on different data, and a clear rejection of
unsupported inputs. Verify configured values reach the underlying estimator
and that plotting produces a readable image without changing the fitted model.
Continue with :doc:`evaluation` for held-out assessment and :doc:`scientific`
for reporting the methods and settings used in a study.
