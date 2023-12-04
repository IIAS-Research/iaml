import sys
import pandas as pd
import numpy as np
sys.path.insert(0, '../')


## AUTOMED ##
from automed import *

dataset = Dataset(pd.read_csv('/Users/rudy/Documents/CHU/iias/baiddy_group/automl/src/perf_logger/tests_data/fetal_health.csv', sep=";"))
dataset.set_label("label")

automed = AutoMed(dataset)
automed.debug_load()
output = automed.run()


###############

print('----------')
print('- OUTPUT -')
print('----------')

out = automed.output[0]
print('model : ', out.model)
print('accuracy : ', out.evaluate())