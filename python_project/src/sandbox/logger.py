from ..automed import AutoMed, Dataset
import pandas as pd

titanic = pd.read_csv('./python_project/src/perf_logger/tests_data/titanic.csv', delimiter=';')

dataset = Dataset(titanic)
dataset.set_label('label')

autom = AutoMed(dataset=dataset)
autom.debug_load()
autom.run()
