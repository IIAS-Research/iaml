import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from automed import TabularClassifier, Dataset

print(TabularClassifier.available_steps())

tc = TabularClassifier()
print(tc.pipeline)


import pandas as pd
import numpy as np

df = pd.read_csv('./data/test.csv', sep=";")
dataset = Dataset(df)

# print(df)

tc.run(dataset)

import random
print(tc.pipeline.first_step.next_steps[0].outputs)
for output in tc.pipeline.outputs():
    rand = random.randint(0,1000)
    print(rand)
    output.data.to_csv('output_'+str(rand)+'.csv')
