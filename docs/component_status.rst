======================
Component availability
======================

Use this catalogue to find an existing component and its configuration
parameters. The linked API categories describe the shipped implementations.
Registration makes a method discoverable; its ``suitable`` check still
determines whether it can run on the task and data at its position.

See :doc:`pipelines/guide` to select training components and
:doc:`pipelines/study` to select analytical components. Add a method missing
from the catalogue with :doc:`adaptability`.

Availability at a glance
========================

.. list-table::
   :header-rows: 1
   :widths: 27 20 53

   * - Capability
     - Status
     - Scope
   * - Training and analytical families below
     - Implemented
     - Included in IAML; individual methods require compatible data and
       prediction outputs.
   * - ``LightGBMClassifier`` and ``LightGBMRegressor``
     - Backend dependent
     - Available when ``lightgbm`` imports successfully. The adapters are
       excluded by suitability checks otherwise.
   * - ``WeibullAFT``
     - Backend dependent
     - Uses a compatible scikit-survival ``WeibullAFT`` implementation or
       falls back to ``lifelines.WeibullAFTFitter``. Requires one of these
       backends to be available.
   * - Advanced domain types, a constraints DSL and partial replacement of
       complete subpipelines
     - Unsupported
     - Outside the current recipe API. See :ref:`pipelines-ref-open` for the
       supported boundaries.

``iaml.steps`` exposes readable names for IAML's training adapters, including
those above. The historical ``Act...`` imports refer to the same classes.

Prepare data
============

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API category
     - What it provides
   * - :doc:`Initial data preparation <autoapi/iaml/actionables/features_precleaning/index>`
     - Standardize formats before modeling, such as trimming whitespace,
       parsing dates, converting numeric strings and recognizing missing-value
       codes.
   * - :doc:`Cleaning and encoding <autoapi/iaml/actionables/cleaning/index>`
     - Handle missing values and convert categorical, textual or date columns
       into model inputs. Includes imputation, category encoders and text
       representations.
   * - :doc:`Feature selection <autoapi/iaml/actionables/features_selection/index>`
     - Retain informative variables and reduce redundant features, using
       variance, correlation, statistical tests or model-based selection.
   * - :doc:`Scaling and normalization <autoapi/iaml/actionables/normalize/index>`
     - Adjust numerical feature scales for models sensitive to units or
       magnitudes. Includes standard, robust and min-max scaling.
   * - :doc:`Class balancing <autoapi/iaml/actionables/imbalance/index>`
     - Change the balance of classes in training data through over- or
       undersampling, including SMOTE. Resampling is skipped during prediction.
   * - :doc:`Feature transformations <autoapi/iaml/actionables/features_preprocessing/index>`
     - Explore alternative representations through dimensionality reduction,
       projections, feature aggregation and distribution transformations.
       Includes PCA and quantile transformations.

Predict outcomes
================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API category
     - What it provides
   * - :doc:`Classification <autoapi/iaml/actionables/predictors/classifier/index>`
     - Predict categorical outcomes, such as disease status or a clinical
       event. Individual estimators determine which target types and
       probability outputs they support.
   * - :doc:`Regression <autoapi/iaml/actionables/predictors/regressor/index>`
     - Predict numerical outcomes, such as a measurement or length of stay,
       using linear models, trees, boosting and other estimator families.
   * - :doc:`Survival analysis <autoapi/iaml/actionables/predictors/survival/index>`
     - Model time-to-event outcomes with censoring. Predictors include Cox
       models, survival forests and survival boosting.

Evaluate and optimize
=====================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API category
     - What it provides
   * - :doc:`Metrics <autoapi/iaml/metrics/index>`
     - Score predictions for classification, regression or survival. The
       main metric defines the objective used to rank candidate pipelines.
   * - :doc:`Validation splitters <autoapi/iaml/splitters/index>`
     - Define training and validation partitions for candidate evaluation,
       using cross-validation or holdout splits. Group information supports
       separation of patients or sites where required by the study.
   * - :doc:`Search optimizers <autoapi/iaml/optimizers/index>`
     - Propose further candidates from previous evaluations. Genetic search
       can change steps and their parameters, while Bayesian and random
       optimizers tune parameters within existing pipeline structures.

Inspect data and results
========================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API category
     - What it provides
   * - :doc:`Descriptive statistics <autoapi/iaml/statistics/index>`
     - Summarize the dataset through missingness, distributions, variability
       and associations. These summaries describe the data rather than
       prediction performance.
   * - :doc:`Plots <autoapi/iaml/plots/index>`
     - Visualize data, model performance and SHAP contributions. Individual
       plot classes specify the task, inputs and prediction outputs they need.
   * - :doc:`Explanation adapters <autoapi/iaml/explainers/index>`
     - Request ``KernelSHAP`` through a configured explanation collection.
       Calculations run on demand on a fitted classification or regression
       pipeline.

See :doc:`architecture` for how these components work together.
