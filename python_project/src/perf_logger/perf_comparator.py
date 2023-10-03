import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
from os.path import exists
current_path = os.path.dirname(os.path.realpath(__file__))

global pick_color_iter
pick_color_iter = -1
def pick_color():
    global pick_color_iter
    pick_color_iter = pick_color_iter + 1    
    return ['red', 'green', 'yellow', 'purple'][pick_color_iter%4]

# Create graph
# df_plot = results.iloc[::-1]
# plot = sns.lineplot(data=df_plot, x="commit_id", y="perf", hue="dataset_name")
# sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
# plot.set(ylim = (.5,1))
# fig = plot.get_figure()
# fig.savefig(current_path+"/../../../docs/perf_fig.png",dpi=300, bbox_inches = "tight") 

history_path = current_path+"/tests_data/history.log"
automed_result = pd.read_csv(current_path+"/tests_data/history.log").iloc[::-1]
autosklearn_results = pd.read_csv(current_path+"/tests_data/autosklearn_perf.log")

datasets = list(automed_result['dataset_name'].unique())
sns.set()
fig, axes = plt.subplots(len(datasets), 1, figsize=(8, len(datasets)*5))

for index, dataset in enumerate(datasets):
    plot = sns.lineplot(data=automed_result[automed_result['dataset_name'] == dataset], x="commit_id", y="perf", hue="dataset_name", ax=axes[index]) 
    plot.set(title=dataset)
    plot.set(ylim = (.5,1.1))
    labels = ['AutoMed']

    for ind, value in autosklearn_results[autosklearn_results['dataset_name'] == dataset].iterrows():
        plot.axhline(y=value['perf'], linestyle='dashed', label="AutoSKLearn "+str(value['time']), color=pick_color())
        labels.append("AutoSKLearn "+str(value['time']))
        
    handles, _ = axes[index].get_legend_handles_labels()
    # Slice list to remove first handle
    plot.legend(handles = handles, labels = labels)
        
    sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
    
    

fig.savefig(current_path+"/../../../docs/compare_perf_fig.png", dpi=300, bbox_inches = "tight") 