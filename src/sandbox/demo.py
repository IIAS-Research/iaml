import sys
import pandas as pd

sys.path.insert(0, './src/iaml')


## AUTOMED ##
from iaml import *

df = pd.read_csv('./src/perf_logger/tests_data/fetal_health.csv', sep=";")
y = pd.DataFrame(df["label"])
X = df.drop(columns=['label'])

iaml = IAML(quiet=True, max_duration=300)
iaml.default_pipeline()
output = iaml.fit(X, y)


###############

print('----------')
print('- OUTPUT -')
print('----------')

out = iaml.output[0]
print('model : ', out.model)
print('accuracy : ', out.evaluate())