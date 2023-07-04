from .tabularstep import TabularStep, Priority
from ...dataset import Dataset


class DropNaStep(TabularStep):
    def __init__(self):
        None
        
    @classmethod
    def priorize(self, dataset: Dataset) -> Priority:
        print("PRIORIZE DROPNA")
        if dataset.data.isna().any().any():
            return Priority.LOW
        else:
            return Priority.NEVER
        
    def run(self, dataset: Dataset) -> Dataset:
        print("RUN DROPNA !!")
        output = Dataset(dataset.data.dropna())
        return output


## Version column by column
# class DropNaStep(TabularStep):
#     def __init__(self, column: str):
#         self.column = column
        
#     @classmethod
#     def priorize(self, dataset: Dataset) -> Priority:
#         if dataset.data.isna().any().any():
#             return Priority.LOW
#         else:
#             return Priority.NEVER
        
#     @classmethod
#     def prepare(self, dataset: Dataset) -> list:
#         return [self(column) for column in dataset.data.columns[dataset.data.isna().any()].tolist()]
        
#     def run(self, dataset: Dataset) -> Dataset:
#         dataset.data.dropna(subset=[self.column])
#         return dataset