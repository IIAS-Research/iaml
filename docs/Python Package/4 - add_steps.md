# Add your own steps

AutoMed is designed to be adaptable and scalable. You can easily create any kind of Step and add it to your pipelines. 

In AutoMed, a step is the smallest componante of a pipeline. There is different types of Steps :
- Actionable : Step that will do an action. For example : Modify Dataset Data, Train a model, Choose a metric
- MetaStep : A MetaStep is a Step that contain other Steps (of any type) and execute them. The default behavior is to execute Steps ordered by priority (each Step computed his own priority).
    - OrderedMetaStep : Contained Step will be executed in order
    - ExplorerMetaStep : All Step will be executed in the same time but independently. This will create one pipeline fork by step in the MetaStep. 
- StepWrapper : A StepWrapper contain only one step and will impact execution. For example, GridSearch is a Wrapper of a learning Step (Actionable)

## How to create a step
There is four easy things to do :
1. Create a class that inherits from the appropriate Step type (Step, Actionable, MetaStep, StepWrapper or also deeper classes)
2. Add class decorators
    1. @is_step(tags,) : This will help AutoMed to know that your Step exist. By adding tags, you can also automatically add your step in existing pipelines. 
    2. @assessable : If your step performances can be evaluated by a metric.
3. Create a constructor (__init__) with your step's configuration and name. 
4. Create a run(input) method with @running decorator and returning an Output instance. @running will carry out all the hard stuff for you (multiple input, multiple output, configurations, etc.).

Here is an basic example of Actionable Step :

```python
def transform(x, y, columns):
    for name, mean in columns:
        x[name] = x[name].fillna(mean)

    return x, y


@is_step('cleaning')
class ActMeanColumn(Actionable):
    def __init__(self):
        self.name = "Fill missing values with mean"
        self.configurations = [{
            'empty_threshold': {
                'description': 'Column with less or equal proportion of empty row will be fill with mean value. 1 will always fill void values',
                'default': 0.5
            }
        }]
    
    @runner
    def run(self, input: Input, callback=None) -> Output:  
        columns = []
        for column in input.dataset.get_columns_names_by_type(DataType.NUMERIC):
            values = input.dataset[column]
            if values.isnull().sum()/len(values) <= self.get_config('empty_threshold'):
                columns.append((column, values.mean()))
        
        return input.transform_dataset(transform, columns)
```

Another example ? Yes ! With a ML model this time :

```python
def learn(model, X):
    return model.predict(X)


@is_step('learning', 'tabular')
class ActRandomForest(Actionable):
    name = "Learn : Random Forest"
    def __init__(self):
        self.configurations = [{
            'max_depth': {
                'description': 'Max depth of each tree',
                'default': 15,
                'range': [1, float('inf')]
            },
            'random_state': {
                'description': 'random_state',
                'default': 42
            }
        }]
        
    @runner
    def run(self, input: TrainingInput, callback=None):
        model = RandomForestClassifier(max_depth=self.get_config('max_depth'), random_state=self.get_config('random_state'))
        
        if input.dataset.is_multilabel:
            model = BinaryRelevance(classifier=model, require_dense=[False, True])
        
        model.fit(input.dataset.X_train, input.dataset.Y_train)
        
        return input.set_model(model, learn)
```

These two Steps was already automatically added to all the pipeline using tags. Easy, isn't it ?