import pandas as pd 
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
from os.path import exists
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_path+"/../")
from automed import *

def each_file(file):
    start_file = datetime.now()
    filename = file.split('/')[-1]
    print('FILE :', filename)
    try:
        df = pd.read_csv(file, sep=";")
    except:
        df = pd.DataFrame()
        
    if len(df.columns) < 2:
        df = pd.read_csv(file, sep=",")
        
    labels = list(set(filter(lambda x: x[0:5] == 'label', df.columns)))

    auto = AutoMed(quiet=True)
    auto.debug_load(fast=False)
    auto.fit(df.drop(labels, axis=1), df[labels])
    
    end_file = datetime.now()
    compute_time = (end_file - start_file).seconds
    
    # Find best result
    max_result = 0
    for output in auto.output:
        tmp = output.get_main_metric_value()
        if tmp > max_result:
            max_result = tmp
    
    return pd.DataFrame([[commit_id, str(time), filename, max_result, compute_time]], columns=results.columns)

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
    threads.append(ThreadWithReturnValue(target=each_file, args=(file,)))
            
# Run all threads
for thread in threads:
    thread.start()
    
# Wait end of all threads
for thread in threads:
    output = thread.join()
    results = pd.concat([output, results], ignore_index=True)

    
results.to_csv(history_path, index=False)
    
# # Create graph
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