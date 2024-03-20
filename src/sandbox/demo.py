import sys
import pandas as pd

sys.path.insert(0, '../')


## AUTOMED ##
from automed import *

df = pd.read_csv('/Users/rudy/Documents/CHU/iias/baiddy_group/automl/src/perf_logger/tests_data/fetal_health.csv', sep=";")
y = pd.DataFrame(df["label"])
X = df.drop(columns=['label'])

automed = AutoMed(quiet=True)
automed.default_pipeline(fast=True)
output = automed.fit(X, y)


###############

print('----------')
print('- OUTPUT -')
print('----------')

out = automed.output[0]
print('model : ', out.model)
print('accuracy : ', out.evaluate())