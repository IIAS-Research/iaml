import pandas as pd


class PredictionDataset:
    def __init__(self, pred_data: pd.DataFrame) -> None:
        self.pred_data = pred_data
    

    def apply(self, method, *args, **kw):
        self.pred_data, _ = method(self.pred_data, [], *args, **kw)
