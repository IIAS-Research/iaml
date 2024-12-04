============
Adaptability
============

IAML is designed to be highly adaptable and extensible, allowing users to seamlessly integrate custom functionality into their machine learning pipelines. Its modular architecture makes it easy to create new Steps and add them to pipelines, ensuring that IAML can be tailored to suit diverse domains and unique project requirements.

Steps
=====

What is a Step?
---------------

In IAML, a **Step** is the smallest unit of a pipeline. Each Step performs a specific task, such as preprocessing data, training a model, or evaluating metrics. Steps are categorized into several types:

- **Actionable Step:** Performs a specific action, such as modifying the dataset, training a model, or computing a metric.
- **MetaStep:** Contains other Steps (of any type) and manages their execution:

  - **OrderedMetaStep:** Executes contained Steps in a specific order.
  - **ExplorerMetaStep:** Executes all contained Steps independently, creating separate pipeline forks for each Step.

Creating a Custom Step
----------------------
Adding custom Steps to IAML pipelines is straightforward. Follow these steps to create a custom Step:

1. **Inherit from the appropriate Step type:** Choose the appropriate base class, such as `Step`, `Actionable`, `MetaStep`, or `Predictor`.
2. **Add required `@is_step(tags)` decorator:** Registers the Step in IAML and associates it with specific pipeline tags. This enables automatic inclusion of the Step in pipelines based on context.
3. **Define a constructor (`__init__`):** Configure your Step by defining its name and parameters. Constructor must define `configuration` dictionary.
4. **Implement the `fit(self, dataset)` method:** This method handles the fitting of steps parameters according current dataset. Must return self.
5. **Implement one of the executions methods:**
    - **transform**: Receive a list of sample (X) and return a transformed version of it (with the sample number of sample).
    - **resample** : Receive a list of sample (X) and labels (y), then return resampled X and y. Warning : This kind of step is mandatory for treatment like RandomUnderSampling but as it received X and y, it must be developed carefully to avoid all biais. 
    - **predict** : Optional if you predictor follow scikit-learn API. Predict method receive a list of samples (X) and return predicted values (y). 

Example: Custom Cleaning Step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here’s an example of an Actionable Step that fills missing values in numeric columns with the mean:

.. code-block:: python
    
    @is_step('cleaning', 'baseline_cleaning')
    class ActMeanColumn(Actionable):
        """[STEP] Fill missing values with the mean."""

        name: str = 'Fill missing values'
        _description: str = textwrap.dedent('''\
            Fill missing values with the mean of non-missing values
            when the proportion of empty rows is lower than {empty_threshold:.0%}.''')
        _description_long: str = textwrap.dedent('''\
            Fill a column missings values with the mean of the columns
            when the proportion of empty rows is lower than {empty_threshold}.
            Work only for numerical columns.''')
        can_be_disabled: bool = False

        def __init__(self):
            self.columns: list[str] = None
            self.configuration:dict = {
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

        name: str = "Linear Regression"
        _description: str = textwrap.dedent('''\
            LinearRegression is a machine learning algorithm that models the
            relationship between input features and a continuous output variable using
            a linear function.''')
        _description_long: str = textwrap.dedent('''\
            LinearRegression is a type of regression algorithm that models
            the relationship between input features and a continuous output variable using
            a linear function. It works by finding the best-fitting line or hyperplane
            that minimizes the sum of the squared differences between the predicted
            and actual output variables.''')
        refs: list[dict[str, Any]] = []

        def __init__(self):
            self.model: LinearRegression = None

        def fit(self, dataset: Dataset):
            self.model = LinearRegression()
            self.model.fit(dataset.X, dataset.y)

            return self

        def suitable(self, dataset: Dataset) -> bool:
            return dataset.type_of_target in ['continuous']

How Tags Enable Automation
--------------------------
Tags allow IAML to automatically add Steps to appropriate pipelines. When you decorate a Step with `@is_step(tags)`, it is registered in IAML with the specified tags. Pipelines can then include the Step dynamically based on context, ensuring that custom Steps integrate seamlessly.

For example:
- A cleaning Step tagged with `'cleaning'` will automatically be included in pipelines where cleaning is required.
- A learning Step tagged with `'learning'` and `'tabular'` will be added to pipelines handling tabular datasets.

Optimizer
==========

TODO

Metrics
=======

TODO

Plots
=====

TODO

Conclusion
==========
IAML’s adaptability makes it a powerful tool for diverse machine learning tasks. With minimal effort, you can create and integrate custom Steps, ensuring that IAML evolves alongside your project requirements.

Explore more about IAML’s architecture in the :doc:`architecture documentation <architecture>`.
