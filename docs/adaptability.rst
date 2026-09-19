============
Adaptability
============

Clinical research studies differ in their data, outcomes and evaluation needs.
IAML exposes preprocessing steps, predictors, metrics and plots as components
that research teams can configure or extend. This page describes how to add
study-specific methods while retaining the same pipeline interfaces.

Steps
=====

What is a Step?
---------------

In IAML, a **Step** is the smallest unit of a pipeline. Each Step performs a specific task, such as preprocessing data, training a model, or evaluating metrics. Steps are categorized into several types:

- **Actionable Step**: Performs a specific action, such as modifying the dataset, training a model, or computing a metric.
- :py:class:`~iaml.metastep.MetaStep`: Contains other Steps (of any type) and manages their execution:

  - :py:class:`~iaml.meta_ordered_step.MetaOrderedStep`: Executes contained Steps in a specific order.
  - :py:class:`~iaml.meta_explorer_step.MetaExplorerStep`: Executes all contained Steps independently, creating separate pipeline forks for each Step.

Creating a Custom Step
----------------------
Adding custom Steps to IAML pipelines is straightforward. Follow these steps to create a custom Step:

1. **Inherit from the appropriate Step type:** Choose the appropriate base class, such as `Step`, `Actionable`, `MetaStep`, or `Predictor`.
2. **Add required `@is_step(tags)` decorator:** Registers the Step in IAML and associates it with specific pipeline tags. This enables automatic inclusion of the Step in pipelines based on context.
3. **Define a constructor (`__init__`):** Configure your Step by defining its name and parameters. Constructor must define ``configuration`` dictionary.
4. **Implement the `fit(self, dataset)` method:** This method handles the fitting of steps parameters according to the current dataset. Must return self.
5. **Implement one of the following methods:**
    - :py:meth:`~iaml.step.Step.transform`: Receives a list of samples (X) and returns a transformed version of it (with the same number of samples).
    - :py:meth:`~iaml.step.Step.resample`: Receives a list of samples (X) and labels (y), then return resampled X and y. Warning: this kind of step is mandatory for processings such as RandomUnderSampling because they need to transform both X and y. However, beware of biases when transforming the labels. 
    - :py:meth:`~iaml.step.Step.predict`: Optional if your predictor follows scikit-learn's API. This method receives a list of samples (X) and returns predicted values (y). 

Example: Custom Cleaning Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here’s an example of an Actionable Step that fills missing values in numeric columns with the mean:

.. code-block:: python
    
    @is_step('cleaning', 'baseline_cleaning')
    class ActMeanColumn(Actionable):
        """[STEP] Fill missing values with the mean."""

        name = 'Fill missing values'
        _description = textwrap.dedent('''\
            Fill missing values with the mean of non-missing values
            when the proportion of empty rows is lower than {empty_threshold:.0%}.''')
        _description_long = textwrap.dedent('''\
            Fill a column missings values with the mean of the columns
            when the proportion of empty rows is lower than {empty_threshold}.
            Work only for numerical columns.''')
        can_be_disabled = False

        def __init__(self):
            self.columns = None
            self.configuration = {
                'empty_threshold': {
                    'description': textwrap.dedent('''\
                        Column with less or equal proportion of empty row will be
                        fill with mean value. 1 will always fill void values'''),
                    'default': 1
                }
            }

        def fit(self, dataset: Dataset) -> Actionable:
            self.columns = []
            explain = []

            for column in dataset.get_columns_names_by_type(DataType.NUMERIC):
                values = dataset.X[column]
                nan_values_count = values.isnull().sum()

                mean = values.mean()
                if np.isnan(mean):
                    mean = 0

                self.columns.append((column, mean))
                explain.append((
                    nan_values_count,
                    len(values),
                    nan_values_count / len(values) * 100,
                ))

            self.explanations = [
                f"""Filled missing values of column **`{c}`** with **{mean:.2f}**
                    (**{v[0]}** out of **{v[1]}** values (**{v[2]:.2f}**%)
                    were missing in train data)."""
                for (c, mean), v in zip(self.columns, explain)
                if v[0] > 0 # hide processings that affected no values
            ]

            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            """Fill NA values with the mean.

            :param pd.DataFrame x: DataFrame to transform.
            :return: Transformed dataset.
            """
            for name, mean in self.columns:
                X[name] = X[name].fillna(mean)

            return X


Example: Custom Model Step
~~~~~~~~~~~~~~~~~~~~~~~~~~
Here’s another example of an Actionable Step that trains a Random Forest classifier:

.. code-block:: python

    @is_step('predictor', 'tabular', 'fast_predictor', 'regressor', 'baseline_predictor')
    class ActLinearRegression(Predictor):
        """[STEP] Linear Regression"""

        name = "Linear Regression"
        _description = textwrap.dedent('''\
            LinearRegression is a machine learning algorithm that models the
            relationship between input features and a continuous output variable using
            a linear function.''')
        _description_long = textwrap.dedent('''\
            LinearRegression is a type of regression algorithm that models
            the relationship between input features and a continuous output variable using
            a linear function. It works by finding the best-fitting line or hyperplane
            that minimizes the sum of the squared differences between the predicted
            and actual output variables.''')
        refs = []

        def __init__(self):
            self.model: LinearRegression = None

        def fit(self, dataset: Dataset):
            self.model = LinearRegression()
            self.model.fit(dataset.X, dataset.y)

            return self

        def suitable(self, dataset: Dataset) -> bool:
            return dataset.type_of_target == 'continuous'

Configuring Survival Models
---------------------------
Configure a survival step before fitting it. The fitted estimator must receive
the configured values, not silently fall back to its own defaults:

.. code-block:: python

    from iaml import ActRandomSurvivalForest, Dataset

    step = ActRandomSurvivalForest()
    step.configure({"n_estimators": 7, "max_depth": 3})
    step.fit(Dataset(X, y))  # y contains (event, time) pairs
    assert len(step.model.estimators_) == 7
    assert step.model.get_params()["max_depth"] == 3

In a step's configuration, ``passthrough=True`` forwards a parameter to the
estimator constructor through ``passthrough_parameters()``. ``passthrough=False``
is reserved for parameters handled explicitly by the step. Discrete search choices
use ``categorical``, rather than ``options``.

Cox, survival forests, survival trees and both gradient boosting implementations
now apply their declared settings. This also applies to IAML's declared defaults,
which can differ from the estimator library's defaults. Historical results may
therefore change after refitting.

Componentwise gradient boosting uses linear components, not trees. Existing
configurations must remove ``max_depth``, ``min_samples_split`` and
``min_samples_leaf`` for this step: those unsupported settings were previously
ignored. ``n_estimators`` and ``learning_rate`` remain configurable.

How Tags Enable Automation
--------------------------
Tags allow IAML to automatically add Steps to appropriate pipelines. When you decorate a Step with `@is_step(tags)`, it is registered in IAML with the specified tags. Pipelines can then include the Step dynamically based on context, ensuring that custom Steps integrate seamlessly.

For example:

- A cleaning Step tagged with `'cleaning'` will automatically be included in pipelines where cleaning is required.
- A learning Step tagged with `'learning'` and `'tabular'` will be added to pipelines handling tabular datasets.

Optimizer
=========

The **optimizer** is responsible for selecting and optimizing the best-performing candidates in a pipeline. IAML uses a genetic algorithm by default. An :py:class:`~iaml.optimizers.optimizer.Optimizer` implements the following:

1. :py:attr:`~iaml.optimizers.optimizer.Optimizer.finished`: Whether the optimizer is done running.
2. :py:meth:`~iaml.optimizers.optimizer.Optimizer.run`: Receives a list of candidates, optimizes these candidates and returns a new list of candidates.

Metrics
=======

**Metrics** are crucial to evaluate the performance of a model on the task that it has been given. IAML implements several metrics which allow you to quickly understand the performance of your pipeline. If you have a specific use case that IAML does not currently support, you can implement your own metric(s). Each :py:class:`~iaml.metric.Metric` implements the following two methods:

1. :py:meth:`~iaml.metric.Metric.compute`: Receives the actual labels and the predicted values from the model, and returns the computed score corresponding to the metric.
2. :py:meth:`~iaml.metric.Metric.suitable`: Receives the training dataset (both the features and the labels) and the type of target to predict (e.g.: `continuous`, `binary`), and tells whether it is relevant to compute the metric.

Example: Custom Metric
----------------------
Metrics maximize their value by default (``greater_is_better = True``).
Set ``greater_is_better = False`` on an error metric to minimize it when selected
as ``main_metric``. Return the original, positive error from ``compute``;
IAML handles the direction for ranking and optimization while keeping raw values
in reports.

This example creates a balanced accuracy metric based on the one of scikit-learn.

.. code-block:: python

    class BalancedAccuracyMetric(Metric):
        """[METRIC] Balanced Accuracy"""

        name = 'Balanced Accuracy'
        _description = textwrap.dedent('''\
            Balanced Accuracy Score is a metric that evaluates a model's 
            performance by considering both positive and negative classes equally. 
            It calculates the average accuracy for each class, making it useful for imbalanced dataset.
            ''')
        _description_long = textwrap.dedent('''\
            Balanced Accuracy Score evaluates how well a predictive 
            model performs, giving equal importance to both positive and negative classes. 
            This is important in healthcare when data is imbalanced.
            To calculate it, you find the accuracy for each class and then average those values. 
            For example, if a model has 70% accuracy for positive cases and 90% for negative cases, 
            the balanced accuracy is (70% + 90%) / 2 = 80%. This metric ensures that the model is effective 
            for all classes, making it valuable for medical decision-making.
            ''')
        refs = [
            {
                'year': 2010,
                'name': 'The Balanced Accuracy and Its Posterior Distribution',
                'authors': [
                    'Kay Henning Brodersen',
                    'Cheng Soon Ong',
                    'Klaas Enno Stephan',
                    'Joachim M. Buhmann'
                ],
                'doi': 'https://doi.org/10.1109/ICPR.2010.764',
                'publisher': textwrap.dedent("""\
                    Proceedings of the 20th International Conference on Pattern Recognition, 3121-24.
                    """)
            },
            {
                'year': 2015,
                'name': textwrap.dedent("""\
                    Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, 
                    Worked Examples, and Case Studies.
                    """),
                'authors': [
                    'John D. Kelleher',
                    'Brian Mac Namee',
                    'Aoife D\'Arcy'
                ],
                'doi': None,
                'publisher': textwrap.dedent("""\
                    Fundamentals of Machine Learning for Predictive Data Analytics: Algorithms, Worked Examples, and Case Studies
                    """)
            }
        ]

        def __str__(self):
            return 'balanced_accuracy'

        def suitable(self, X: pd.DataFrame, y: pd.DataFrame, type_of_target: str) -> bool:
            return type_of_target in ['binary', 'multiclass']

        def compute(self, y: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> float:
            return balanced_accuracy_score(y, y_pred)

Plots
=====

**Plots** are a great way to vizualise your metrics and the performance of your model. IAML already implements a large variety of plots; they offer a simple API to use and to build on so that you can focus on improving your pipeline. Their API is actually very similar to those of metrics, making it easy for you to implement your own plots. Each :py:class:`~iaml.plot.Plot` (or :py:class:`~iaml.metric_plot.MetricPlot`) implements the following two methods:

1. :py:meth:`~iaml.plot.Plot.compute`: Receives the estimator and the training dataset (both the features and the labels), computes values to vizualise and generates the plot.
2. :py:meth:`~iaml.plot.Plot.suitable`: Receives the type of target to predict (e.g.: `continuous`, `binary`), and tells whether it is relevant to generate the plot.

Example: Custom Metric Plot
---------------------------
This example creates a metric plot which generates the ROC curve.

.. code-block:: python

    class ROCAUCPlot(MetricPlot):
        """[PLOT] ROC-AUC Plot"""

        title = "Receiver Operating Characteristic - Area Under the Curve"
        description = textwrap.dedent("""
            The ROC-AUC (Receiver Operating Characteristic - Area Under the Curve) plot is a widely used 
            tool to assess the performance of a classification model, especially in the healthcare domain. 
            It provides a graphical representation of the model's ability to distinguish between classes, 
            such as diagnosing the presence or absence of a medical condition.
            """)

        @capture
        def compute(
            self,
            estimator: IAMLPipeline,
            X: pd.DataFrame,
            y: pd.Series,
            X_train: pd.DataFrame = None,
            y_train: pd.Series = None,
            **kwargs) -> MetricPlot:
            self._binary_image = io.BytesIO()

            pos_label = None
            if y.dtype not in ['int', 'bool']:
                pos_label = y.iloc[0] if isinstance(y, pd.Series) else y[0]

            # Predict probabilities
            y_prob = estimator.predict_proba(X)[:, 1]

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y, y_prob, pos_label=pos_label)
            roc_auc = auc(fpr, tpr)

            # Create the ROC plot
            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random guess')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc='lower right')
            plt.grid(True)

            # Save plot to binary image
            plt.savefig(self._binary_image, format='png')

            return self

        @classmethod
        def suitable(cls, type_of_target: str) -> bool:
            return type_of_target == 'binary'

Related documentation
=====================
See :doc:`architecture` for how components form a pipeline and :doc:`scientific`
for how to describe the methods used in a study.
