"""
    Benchmark automl lib. Only NaiveAutoML for now
"""
import sys, os, time
import dill as pickle
import traceback
import glob
from datetime import datetime
from os.path import exists
import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import LabelEncoder

from fedot.api.main import Fedot
import naiveautoml
import tpot
from flaml import AutoML as flamlAutoMl

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

def each_file(file:str, package, duration) -> pd.DataFrame:
    """
    Fit and evaluate automl model on a file
    """
    package_name, trainer = package
    filename = file.split('/')[-1]
    
    # TRAIN
    folds_results = []
    try:
        for idx, (X, y, X_test, y_test) in enumerate(file_to_fold(file)):
            if already_computed(filename, duration, package_name):
                print("Already computed -> next.")
                continue
            
            start_fold = time.time()
            
            print('#####')
            print(f'##### {package_name} FOR {duration}s ON {filename} | fold {idx}')
            print('#####')
            
            model, predict_method, model_name = trainer(X, y, duration)

            # EVALUATE
            y_pred = predict_method(X_test)
            fold_dict = {"bench_date": start_time,
                        "package": package_name,
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
            save(model, f"{package_name}_{duration}_{idx}_{filename}")
    except Exception as ex:
        print(traceback.format_exc())
        print('ERROR with', filename)
        print(ex)
        return []
    
    return folds_results

def save(estimator, name):
    file = open(f"{CURRENT_PATH}/pickle_bench/{name}.dill", 'wb')
    pickle.dump(estimator, file)
    file.close()

start_time = int(datetime.now().timestamp())
results = []

def already_computed(dataset, duration, package):
    for result in results:
        if result['dataset'] == dataset and result['max_duration'] == duration and result['package'] == package:
            return True
        
    return False


def train_automed(X, y, duration):
    Cache.reset()
    estimator = AutoMed(quiet=True, max_workers=12, max_duration=duration)
    estimator.fit(X, y)
    return estimator.chosen_model, estimator.chosen_model.predict, f"{estimator.chosen_model.transformers[-1][0]} -> {estimator.chosen_model.predictor[0]}"

def train_naive(X, y, duration):
    estimator = naiveautoml.NaiveAutoML(timeout=duration)
    estimator.fit(X, y)
    return estimator.chosen_model, estimator.chosen_model.predict, f"{estimator.chosen_model}"

def train_fedot(X, y, duration):
    problem = 'classification' if type_of_target(y) in ['binary', 'multiclass'] else 'regression'
    estimator = Fedot(problem=problem, timeout=duration/60.0, preset='best_quality', n_jobs=12)
    estimator.fit(features=X, target=y)
    return estimator, estimator.predict, f"{estimator.current_pipeline}"

def train_tplot(X, y, duration):
    encoder = None
    if type_of_target(y) in ['binary', 'multiclass']:
        class_estimator = tpot.TPOTClassifier
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
    else:
        class_estimator = tpot.TPOTRegressor
        
    estimator = class_estimator(max_time_mins=duration/60, n_jobs=12)
    X = X.drop(columns=X.select_dtypes(include=['object']).columns)
    estimator.fit(X, y)
    
    def predict(X):
        X = X.drop(columns=X.select_dtypes(include=['object']).columns)
        if not encoder:
            return estimator.predict(X)
        else:
            return encoder.inverse_transform(estimator.predict(X))
        
    return estimator, predict, f"{estimator}"

def train_flaml(X, y, duration):
    problem = 'classification' if type_of_target(y) in ['binary', 'multiclass'] else 'regression'
    estimator = flamlAutoMl()
    estimator.fit(X, y, task=problem, time_budget=duration, log_level='error')
    return estimator, estimator.predict, f"{estimator}"

packages = [
    ('naive_autoML', train_naive),
    ('FEDOT', train_fedot),
    ('automed', train_automed),
    # ('tplot', train_tplot),
    ('flaml', train_flaml)
]

def main():
    global results
    
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

    durations = [30, 120, 300, 900, 1800]
    for duration in durations:
        for file in files:        
            for current_package in packages:
                if current_package[0] == 'FEDOT' and file.split('/')[-1] in ["IMDB-Dataset.csv", "bbc-text.csv"]:
                    continue
                
                print(str(datetime.now()), "->>>", file)
                try:
                    outputs = each_file(file, current_package, duration)

                    results += outputs
                    pd.DataFrame(results).to_csv(history_path, index=False)
                except Exception as e:
                    print(traceback.format_exc())
                    print("FAIL : ", file)
                    print(f"ERROR : {e}")
                    time.sleep(10)

if __name__ == "__main__":
    main()