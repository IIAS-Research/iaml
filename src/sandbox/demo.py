import sys
import pandas as pd

sys.path.insert(0, '../')


## AUTOMED ##
from iaml import *

df = pd.read_csv('/Users/rudy/Documents/CHU/iias/baiddy_group/automl/src/perf_logger/tests_data/fetal_health.csv', sep=";")
y = pd.DataFrame(df["label"])
X = df.drop(columns=['label'])

iaml = IAML(quiet=True)
iaml.default_pipeline(fast=True)
output = iaml.fit(X, y)


###############

print('----------')
print('- OUTPUT -')
print('----------')

out = iaml.output[0]
print('model : ', out.model)
print('accuracy : ', out.evaluate())