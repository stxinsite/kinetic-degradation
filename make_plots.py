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


"""
Degradation over time
"""
result_id = 'PROTAC1&ACBI1_fix_BPD_ec=0.0005_vary_time_DEG'
result = pd.read_csv(f"./saved_objects/{result_id}.csv")
result = result[['t', 'degradation', 'all_Ternary', 'Dmax', 'PROTAC']]
result = result.rename(columns={'degradation': 'Degradation', 'all_Ternary': 'Ternary complex'})
result = result.melt(id_vars=['PROTAC', 't'])
# result = result.rename(columns = {'variable': 'Measure'})

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='t',
    y='value',
    hue='PROTAC',
    style='variable',
    palette='Set2',
    ax=ax
)
p.tick_params(labelsize=15)
plt.xlim(result['t'].min(), result['t'].max())
plt.ylim(-5, 120)
plt.xlabel('Time (h)')
plt.ylabel('% Baseline Target Protein')
plt.title(r'Target Protein Degradation with initial $[PROTAC]_{ec} = 0.0005 \mu$M')
handles, labels = ax.get_legend_handles_labels()
labels[3] = "\nvariable"
ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"./plots/{result_id}.png", bbox_inches="tight")

# total BPD_ic and Target over time
result = pd.read_csv(f"./saved_objects/{result_id}.csv")
result.columns
result = result[['t', 'PROTAC', 'total_target', 'total_target_ub', 'total_bpd_ic', 'total_naked_ternary', 'total_ternary']]
result = result.melt(id_vars=['t', 'PROTAC'])

sns.set_style("whitegrid")
fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True)  # break y-axis into two portions
fig.subplots_adjust(hspace=0.15)  # adjust space between axes
# plot same data on both axes
p = sns.lineplot(
    data=result,
    x='t',
    y='value',
    hue='variable',
    style='PROTAC',
    palette='Set2',
    ax=ax
)
q = sns.lineplot(
    data=result,
    x='t',
    y='value',
    hue='variable',
    style='PROTAC',
    palette='Set2',
    ax=ax2,
    legend=False
)
r = sns.lineplot(
    data=result,
    x='t',
    y='value',
    hue='variable',
    style='PROTAC',
    palette='Set2',
    ax=ax3,
    legend=False
)
r.tick_params(labelsize=15)
# limit the view to different portions of data
ax.set_ylim(0.125e-13, 4e-13)  # total target
ax2.set_ylim(0, 0.125e-13)  # total bpd_ic
ax3.set_ylim(0, 1e-15)
ax3.set_xlim(result['t'].min(), result['t'].max())
# hide ylabels
ax.set(ylabel=None, xlabel=None)
ax2.set(ylabel=None, xlabel=None)
ax3.set(ylabel=None, xlabel=None)
# hide the spines between ax and ax2
ax.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax3.xaxis.tick_bottom()
plt.xticks(np.arange(start=0, stop=24 + 1, step=4))
# cut-out slanted lines
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
ax2.plot([0, 1, 0, 1], [0, 0, 1, 1], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)

fig.supxlabel('Time (h)')
fig.supylabel('Amount (umol)')
fig.suptitle(r'Target Protein Degradation with initial $[PROTAC]_{ec} = 0.0005 \mu$M')

handles, labels = ax.get_legend_handles_labels()
labels[1] = "total Target"
labels[2] = "total Ubiquitinated Target"
labels[3] = "total intracellular PROTAC"
labels[4] = "total naked Ternary complex"
labels[5] = "total Ternary complex"

labels[6] = "\n"
ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"./plots/{result_id}_ic_amounts.png", bbox_inches="tight")


"""
Permeability
"""
PROTAC_ID = 'SiTX_38406'

result = pd.read_csv(f"saved_objects/{PROTAC_ID}_Permeability_DEG.csv")
PS_cell_range = result['PS_cell'].unique()
result.head()
result.shape
result = result[['t', 'degradation', 'Ternary', 'all_Ternary', 'PS_cell']]
result_melt = result.melt(id_vars = ['PS_cell', 't'])
result_melt.head()

sns.set_style("whitegrid")
p = sns.lineplot(
    data = result_melt,
    x = 't',
    y = 'value',
    hue = 'PS_cell',
    style = 'variable',
    palette = 'Set2'
)
plt.xlim(result['t'].min(), result['t'].max())
plt.ylim(-5, 120)
plt.xlabel('Time (h)')
plt.ylabel('% Baseline Target Protein')
plt.title(f'Degradation with initial [{PROTAC_ID}] = 0.1uM')
plt.savefig(f"plots/{PROTAC_ID}_Permeability_DEG.png")
