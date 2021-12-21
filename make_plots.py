import numpy as np
import pandas as pd
import seaborn as sns
from sigfig import round
import matplotlib.pyplot as plt
import json
import multiprocessing
from src.kinetic_module.calc_full_config import KineticParameters
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (10,7)

"""
Degradation and Dmax vs PROTAC concentration
"""
result = pd.read_csv("./saved_objects/PROTAC1&ACBI1_vary_BPD_ec_fix_time_DEG.csv")
result.head()
result = result[['initial_BPD_ec_conc', 'degradation', 'Dmax', 'PROTAC']]
result = result.rename(columns = {'degradation': 'Degradation'})
result = result.melt(id_vars = ['PROTAC', 'initial_BPD_ec_conc'])
result.head()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='initial_BPD_ec_conc',
    y='value',
    hue='PROTAC',
    style='variable',
    palette='Set2',
    ax=ax
)
p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(result['initial_BPD_ec_conc'].min(), result['initial_BPD_ec_conc'].max())
plt.ylim(-5, 120)
plt.xlabel(r'Initial $[PROTAC]_{ec}$ $(\mu M)$')
plt.ylabel('% Baseline Target Protein')
plt.title(r'Target Protein Degradation and Dmax at $t = 6h$')
handles, labels = ax.get_legend_handles_labels()
labels[3] = "\nVariable"
ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.savefig(f"./plots/PROTAC1&ACBI1_vary_BPD_ec_fix_time_DEG.png", bbox_inches="tight")

