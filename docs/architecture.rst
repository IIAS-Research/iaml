============
How it works
============

IAML combines data preparation, model search and evaluation in a modular
workflow for tabular clinical research. Each pipeline is assembled from
inspectable steps, allowing research teams to review the methods used and adapt
components to their study.

Learning Architecture
=====================
The IAML framework operates in three main stages:

1. **Candidate Generation:** Initial pipelines are created based on available modules.
2. **Candidate Evaluation:** Pipelines are evaluated for performance using cross-validation.
3. **Candidate Optimization:** The configured optimizer refines the candidates; genetic search is the default.

Each stage is modular, allowing users to customize, extend, or replace components for specific tasks or datasets.

Modular Components
-------------------
IAML uses a set of modular units called "Steps". Each Step represents a specific data processing action, such as data cleaning, feature engineering, or model training. These Steps are assembled into pipelines that are optimized for performance.

Key Features of a Step:

- **Capability Evaluation:** Determines its suitability for the dataset.
- **Parameter Adaptation:** Adjusts its internal parameters for the dataset.
- **Data Action:** Performs an action, such as transformation, prediction, or resampling.
- **Explainability:** Provides a description of operations performed on the data.

Candidate Generation
---------------------
The first stage of IAML’s workflow involves generating initial pipeline candidates. Each candidate is an ordered sequence of Steps, where the output of one Step is used as the input for the next. The pipeline creation process includes:

- Querying all available Steps for suitability.
- Sequentially adding Steps to the pipeline (e.g., imputing missing values, encoding categorical data).
- Generating multiple candidate pipelines tailored to the problem type and dataset.

Candidate Evaluation
---------------------
In the second stage, each candidate is assessed using **5-fold cross-validation**
by default. Performance metrics are averaged across folds to compare candidates
under the same validation strategy.

The candidates are ranked based on a key metric defined by the user (e.g., accuracy for classification or R² for regression). This ranking determines which candidates move on to the optimization stage.

.. note::

    The number of folds in the cross-validation is fixed to **5** by default but, as with everything in IAML, it can be changed.
    If you plan to optimize pipelines for longer durations (e.g., more than 30 minutes), consider increasing this value using IAML’s parameters. 

Shared Evaluation Cache
-----------------------
Training and evaluation caches identify a dataset by its features, targets,
groups, column types and target type. The fingerprint is recomputed from the
current contents, so changing labels or groups cannot reuse results from the
previous dataset. Keys are captured before preprocessing can mutate the data.

Cached preprocessing restores both the fitted step and its training dataset,
preserving shared references needed by transformations such as target encoding.
The shared cache returns copies so later transformations cannot modify its entries.

Validation partitions also depend on the splitter and its parameters. Cached
evaluation scores additionally depend on the metrics and their configuration.
Custom splitters must be deterministic for the same inputs and settings. When
their configuration cannot be serialized (for example, a local lambda), partition
and score caching is bypassed; evaluation still runs normally.

Candidate Optimization
----------------------
IAML’s default optimization process is inspired by **genetic algorithms**. This stage iteratively refines the pipelines by:

- **Selection:** Retaining top-performing candidates from the evaluation stage.
- **Mutation:** Slightly altering the parameters of existing candidates to explore variations.
- **Random Generation:** Introducing new random candidates to encourage exploration.

The default optimizer explores changes to pipeline structure and parameters.
Evaluation and optimization continue until a stopping condition is met, such as
the search time limit or a configured patience limit. Alternative optimizers
include Bayesian hyperparameter tuning; see :doc:`adaptability` for the
optimizer interface.

Learning Overview
=================
IAML’s training process begins with candidate generation: IAML creates multiple
pipelines based on available modules, forming the initial pool of candidates. Next,
it enters a loop of evaluation and optimization. Each candidate in the pool is evaluated
using 5-fold cross-validation. Based on these evaluations, a new generation of candidates
is produced using the configured optimizer. This loop continues until a stopping condition
is met, such as reaching a set patience limit or a timeout.

.. figure:: architecture_flow_diagram.png
    :alt: Architecture flow diagram of IAML
    :align: center
    :width: 80%



Explainability and Transparency
===============================
Clinical researchers and data scientists can inspect the selected pipeline and
request outputs to support interpretation and study reporting:

- **Descriptive statistics and visualizations:** Automatically generated statistics and visualizations to help users understand the input data.
- **Pipeline summaries:** Detailed descriptions of all transformations and models used in a pipeline.
- **Performance metrics:** Comprehensive evaluation metrics for each candidate pipeline.
- **Model performance visualization:** Task-specific plots of predictions and errors on an evaluation dataset.
- **Feature importance:** Feature importance explanations using SHAP (SHapley Additive exPlanations).

Learn how to use all these functionalities in the :doc:`explainability documentation <explainability>`.

Adaptability
============
Research teams can adapt IAML to their data and study design. Users can:

- Modify preprocessing Steps, evaluation metrics, and optimization strategies with minimal effort.
- Integrate new methods or algorithms by implementing custom Steps.
- Leverage GPU acceleration through libraries like cuDF for faster processing.

See the :doc:`adaptability documentation <adaptability>` for component and
extension APIs.
