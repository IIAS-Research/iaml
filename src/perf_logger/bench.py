"""
    Benchmark automl lib. Only NaiveAutoML for now
"""
import sys, os, time
import traceback
import glob
from datetime import datetime
from os.path import exists
import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target

import naiveautoml

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_PATH+"/../")
from automed.automed import *

def file_to_fold(file_path:str):
    """
    Load a file and split it to X, y, X_test, y_test 
    """
    try:
        df = pd.read_csv(file_path, sep=";")
    except Exception:
        df = pd.DataFrame()
        
    if len(df.columns) < 2:
        df = pd.read_csv(file_path, sep=",")
                
    y = np.array(df['label'])
    X = df.drop(columns=['label'])
    
    if type_of_target(y) in ['binary', 'multiclass']:
        kfold = StratifiedKFold(5)
    else:
        kfold = KFold(5)
    
    for idx_train, idx_test in kfold.split(X, y):
        X_train = X.iloc[idx_train].copy()
        X_test = X.iloc[idx_test].copy()
        y_train = y[idx_train].copy()
        y_test = y[idx_test].copy()
        
        yield (X_train, y_train, X_test, y_test)

def each_file(file:str, trainer, duration) -> pd.DataFrame:
    """
    Fit and evaluate automl model on a file
    """
    
    filename = file.split('/')[-1]
    
    # TRAIN
    folds_results = []
    try:
        for idx, (X, y, X_test, y_test) in enumerate(file_to_fold(file)):
            start_fold = time.time()
            
            print('FILE :', filename, 'fold', idx)
            model, model_name = trainer(X, y, duration)

            # EVALUATE
            y_pred = model.predict(X_test)
            fold_dict = {"bench_date": start_time,
                        "fold": idx,
                        "max_duration": duration,
                        "compute_time": int(time.time() - start_fold),
                        "dataset": filename,
                        "model_name": model_name}
            
            for metric_sub_class in Metric.__subclasses__():
                metric = metric_sub_class()
                if metric.suitable(X, y, type_of_target(y)):
                    fold_dict[str(metric)] = metric.compute(y_test, y_pred)
            
            folds_results.append(fold_dict)
    except Exception as ex:
        print(traceback.format_exc())
        print('ERROR with', filename)
        print(ex)
        return []
    
    return folds_results

start_time = int(datetime.now().timestamp())

def train_automed(X, y, duration):
    Cache.reset()
    estimator = AutoMed(quiet=True, max_workers=12, max_duration=duration)
    estimator.fit(X, y)
    return estimator.chosen_model, f"{estimator.chosen_model.transformers[-1][0]} -> {estimator.chosen_model.predictor[0]}"


def train_naive(X, y, duration):
    estimator = naiveautoml.NaiveAutoML(timeout=duration)
    estimator.fit(X, y)
    return estimator.chosen_model, f"{estimator.chosen_model}"


def main():
    durations = [30, 120, 300, 900, 1800]
    for duration in durations:
        files = glob.glob(CURRENT_PATH+"/tests_data/*.csv")
        dont_push_csv = glob.glob(CURRENT_PATH+"/tests_data/dont_push/*.csv")
        files = files + dont_push_csv
        
        history_path = CURRENT_PATH+"/tests_data/bench_v2.log"

        if not exists(history_path):
            if not exists(CURRENT_PATH+"/tests_data/"):
                os.mkdir(CURRENT_PATH+"/tests_data/")
            results = []
        else:
            results = pd.read_csv(history_path).to_dict('records')


        for file in files:
            print(str(datetime.now()), "->>>", file)
            try:
                outputs = each_file(file, train_naive, duration)

                results += outputs
                pd.DataFrame(results).to_csv(history_path, index=False)
            except Exception as e:
                print(traceback.format_exc())
                print("FAIL : ", file)
                print(f"ERROR : {e}")

if __name__ == "__main__":
    main()