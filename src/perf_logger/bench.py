"""
    Benchmark automl lib. Only NaiveAutoML for now
"""
import sys, os
import glob
from datetime import datetime
from os.path import exists
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import balanced_accuracy_score

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, CURRENT_PATH+"/../")
from automed import *

# current_duration = 30 #60*15
# current_duration = int(sys.argv[1])

def load_file(file_path:str):
    """
    Load a file and split it to X, y, X_test, y_test 
    """
    try:
        df = pd.read_csv(file_path, sep=";")
    except Exception:
        df = pd.DataFrame()
        
    if len(df.columns) < 2:
        df = pd.read_csv(file_path, sep=",")
                
    labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=43)
    y = np.array(train_df[labels[0]])
    X = train_df.drop(columns=labels)
    
    y_test = np.array(test_df[labels[0]])
    X_test = test_df.drop(columns=labels)
        
    return X, y, X_test, y_test

def each_file(file:str, duration) -> pd.DataFrame:
    """
    Fit and evaluate automl model on a file
    """
    start_file = datetime.now()
    
    filename = file.split('/')[-1]
    print('FILE :', filename)
    X, y, X_test, y_test = load_file(file)

    Cache().reset()
    estimator = AutoMed(quiet=True, max_workers=12, max_duration=duration)
    estimator.fit(X, y)
    
    end_file = datetime.now()
    compute_time = (end_file - start_file).seconds
    
    y_pred = estimator.chosen_model.predict(X_test)
    if type_of_target(y_test) == 'continuous':
        best_result = r2_score(y_test, y_pred)
    else:
        best_result = balanced_accuracy_score(y_test, y_pred) 
    
    return pd.DataFrame([
            [duration, str(estimator.chosen_model.predictor[1].__class__), str(time), filename, best_result, compute_time]
        ]
        , columns=['duration', 'model_name', 'date', 'dataset_name','perf', 'compute_time'])

time = int(datetime.now().timestamp())

def main():
    durations = [30, 120, 300, 900, 1800]
    while True:
        for duration in durations:
            files = glob.glob(CURRENT_PATH+"/tests_data/*.csv")
            dont_push_csv = glob.glob(CURRENT_PATH+"/tests_data/dont_push/*.csv")
            files = files + dont_push_csv

            history_path = CURRENT_PATH+"/tests_data/bench_automed2_2.log"

            if not exists(history_path):
                if not exists(CURRENT_PATH+"/tests_data/"):
                    os.mkdir(CURRENT_PATH+"/tests_data/")
                results = pd.DataFrame(columns= \
                    ['duration', 'model_name', 'date', 'dataset_name','perf', 'compute_time'])
            else:
                results = pd.read_csv(history_path)


            # files = [files[5]]
            for file in files:
                print("->>>", file)
                print("TIME", str(datetime.now()))
                
                try:
                    output = each_file(file, duration)
                    
                    print("######")
                    print("RESULT", file, output)
                    print("######")
                    
                    results = pd.concat([output, results], ignore_index=True)
                    results.to_csv(history_path, index=False)
                except Exception as e:
                    print("FAIL : ", file)
            
    print("FINISHED")

if __name__ == "__main__":
    main()