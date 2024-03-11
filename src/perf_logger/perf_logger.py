import pandas as pd 
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


import sys, os
from os.path import exists
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_path+"/../")
from automed import *

DURATION = 60*30 # 30 minutes

def each_file(file):
    Logger.reset()
    Cache.reset()
    start_file = datetime.now()
    filename = file.split('/')[-1]
    print('FILE :', filename)
    try:
        df = pd.read_csv(file, sep=";")
    except:
        df = pd.DataFrame()
        
    if len(df.columns) < 2:
        df = pd.read_csv(file, sep=",")
        
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=43)

    labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))
    y = train_df[labels]
    X = train_df.drop(columns=labels)

    y_test = test_df[labels]
    X_test = test_df.drop(columns=labels)
        

    auto = AutoMed(quiet=True)
    auto.default_pipeline(fast=False)
    local_results = auto.fit(X, y, max_duration=DURATION, patience=5)
    
    end_file = datetime.now()
    compute_time = (end_file - start_file).seconds
    
    best_result = local_results[0].evaluate(X_test, y_test)[local_results[0].main_metric]
    
    return pd.DataFrame([[commit_id, str(time), filename, best_result, compute_time]], columns=results.columns)

commit_id = sys.argv[1]
time = int(datetime.now().timestamp())

files = glob.glob(current_path+"/tests_data/*.csv")
dont_push_csv = glob.glob(current_path+"/tests_data/dont_push/*.csv")

files = files + dont_push_csv

history_path = current_path+"/tests_data/history.log"
results = None
if not exists(history_path):
    if not exists(current_path+"/tests_data/"):
        os.mkdir(current_path+"/tests_data/")
    results = pd.DataFrame(columns=['commit_id', 'date', 'dataset_name','perf', 'compute_time'])
else:
    results = pd.read_csv(history_path)

threads = []
        
# Create one thread by File
# files = [files[5]]

for file in files:
    print("->>>", file)
    print("TIME", str(datetime.now()))
    output = each_file(file=file)
    results = pd.concat([output, results], ignore_index=True)
    results.to_csv(history_path, index=False)
    
# Create graph
df_plot = results.iloc[::-1]
plot = sns.lineplot(data=df_plot, x="commit_id", y="perf", hue="dataset_name")
sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
# plot.set(ylim = (.5,1))
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
fig = plot.get_figure()
fig.savefig(current_path+"/../../docs/perf_fig.png",dpi=300, bbox_inches = "tight") 

# Create time graph
plt.clf()
plot = sns.lineplot(data=df_plot, x="commit_id", y="compute_time", hue="dataset_name")
sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
plot.set_xticklabels(plot.get_xticklabels(), rotation=90)
fig = plot.get_figure()
fig.savefig(current_path+"/../../docs/time_fig.png",dpi=300, bbox_inches = "tight") 