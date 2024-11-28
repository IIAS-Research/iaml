# Quick Start

Automed was created to make machine learning as simple as possible. So you will need very few lines of python to get a AI model. 

```python
# First you need to import the package in your environment and pandas.
import automed
import pandas as pd

# Then you will import your data
your_data = pd.from_csv("path/to/your/data.csv")

# Create an instance of AutoMed with data
autom = AutoMed(dataset=your_data)

# Tell AutoMed that you want classification
autom.load_classifier_pipe()

# Finaly run the pipe !
results = autom.run()

# You will get several results ordered by the quality of the predictions. 
# Each result is an instance of "Output" and will contain three attributes -> dataset, model, metric
# - dataset -> Dataset with train and test set which was used for the training and testing
# - model -> AI model that you can use to make predictions
# - metric -> metrics uses to evaluate the models 
print(results[0].dataset.X_train)
print(results[0].model)
print(results[0].metric)

# You can re-computed metrics with evaluate function
print(results[0].evaluate())

# That's all !
```

That's all for the quick start. With it you will bee able to have models with very good result but you can get a bit further and discover how to get even better results by finetuning the pipeline with your medicals knowledges. 