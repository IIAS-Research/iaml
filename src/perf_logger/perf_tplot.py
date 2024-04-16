import pandas as pd 
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import sys, os
from os.path import exists
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_path+"/../")
from iaml import *


files = glob.glob(current_path+"/tests_data/dont_push/*.csv") + glob.glob(current_path+"/tests_data/*.csv")

history_path = current_path+"/tests_data/tplot_perf.log"
results = None
if not exists(history_path):
    if not exists(current_path+"/tests_data/"):
        os.mkdir(current_path+"/tests_data/")
    results = pd.DataFrame(columns=['time', 'dataset_name','perf'])
else:
    results = pd.read_csv(history_path)

# files = []
for time in [60]: #[30, 60, 60*5, 60*20]:
    for file in files:
        filename = file.split('/')[-1]
        print('FILE :', filename, 'TIME :', str(time/60.0))
        df = pd.read_csv(file, sep=";")
        if len(df.columns) < 2:
            df = pd.read_csv(file, sep=",")
            
        auto = IAML(Dataset(df))
        auto.dataset.set_label('label') 
        auto.tplot_load()
        auto.run()
        
        # Find best result
        max_result = 0
        for output in auto.output:
            tmp = output.evaluate()
            if tmp > max_result:
                max_result = tmp
        
        new_record = pd.DataFrame([[str(time/60.0), filename, max_result]], columns=results.columns)
        results = pd.concat([new_record, results], ignore_index=True)
        results.to_csv(history_path, index=False)
        
    
# results.to_csv(history_path, index=False)
    
# Create graph
# df_plot = results.pivot_table(index='commit_id', columns='dataset_name', values='perf')
# plot = sns.lineplot(data=df_plot)
df_plot = results.sort_values(by=['time']).astype({'time': str})
plot = sns.lineplot(data=df_plot, x="time", y="perf", hue="dataset_name")
sns.move_legend(plot, "upper left", bbox_to_anchor=(1, 1))
# plot.set(ylim = (.5,1))
fig = plot.get_figure()
fig.savefig(current_path+"/../../docs/tplot_perf_fig.png", dpi=300, bbox_inches = "tight") 