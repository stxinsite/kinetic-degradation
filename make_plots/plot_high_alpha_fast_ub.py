import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (10, 7)

protac_id = 'ACBI1'
bpd_ec = 0.001
alpha = 1000
k_ub = 5000

result_id = f"{protac_id}_BPD_ec={bpd_ec}_high_alpha_fast_ub_DEG"
result = pd.read_csv(f"./saved_objects/{result_id}.csv")

important_result = result[['t',
                           'total_target',
                           'total_naked_ternary',
                           'total_poly_ub_ternary',
                           'total_poly_ub_target'
                           ]]
important_result = important_result.melt(id_vars=['t'])

sns.set_style('whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
p = sns.lineplot(data=important_result, x='t', y='value', hue='variable', palette='Set2', ax=ax1)
q = sns.lineplot(data=important_result, x='t', y='value', hue='variable', palette='Set2', ax=ax2)
# limit the view to different portions of data
ax1.set_ylim(0, 4.5e-13)
ax2.set_ylim(0, 2.5e-14)
# hide labels
ax1.set(ylabel=None, xlabel=None)
ax2.set(ylabel=None, xlabel=None)
# hide the spines between ax1 and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False, labelsize=15)
ax2.xaxis.tick_bottom()
ax2.tick_params(labelsize=15)
# set xticks
plt.xticks(np.arange(start=0, stop=25, step=4))
# cut out slanted lines
d = 0.5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
# set labels
fig.supxlabel('Time (h)')
fig.supylabel('Amount (umol)')
fig.suptitle(fr'Initial $[{protac_id}] = {bpd_ec}\mu$M, $\alpha = {alpha}$, ' + r'$k_{ub}$' + fr'$= {k_ub}/h$')
# set legends
handles1, labels1 = ax1.get_legend_handles_labels()
labels1[0] = "Total target"
ax1.legend(handles=handles1[0:1], labels=labels1[0:1], loc='upper right')
handles2, labels2 = ax2.get_legend_handles_labels()
labels2[1] = "Naked ternary"
labels2[2] = "Fully Ub'd ternary"
labels2[3] = "Fully Ub'd target"
ax2.legend(handles=handles2[1:], labels=labels2[1:], loc='upper right')
# set legend text
plt.setp(ax1.get_legend().get_texts(), fontsize='15')
plt.setp(ax2.get_legend().get_texts(), fontsize='15')
# save plot
plt.savefig(f"./plots/{result_id}_amounts.png")


rates = result[['t', 'degradation_rate', 'naked_ternary_rate', 'poly_ub_ternary_rate', 'poly_ub_target_rate']]
rates = rates.melt(id_vars=['t'])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
p = sns.lineplot(data=rates, x='t', y='value', hue='variable', palette='Set2', ax=ax1)
q = sns.lineplot(data=rates, x='t', y='value', hue='variable', palette='Set2', ax=ax2)
# limit the view to different portions of data
ax1.set_ylim(-1e-14, 2e-14)
ax2.set_ylim(-1.5e-13, 0.25e-13)
# hide labels
ax1.set(ylabel=None, xlabel=None)
ax2.set(ylabel=None, xlabel=None)
# hide the spines between ax1 and ax2
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False, labelsize=15)
ax2.xaxis.tick_bottom()
ax2.tick_params(labelsize=15)
# set xticks
plt.xticks(np.arange(start=0, stop=25, step=4))
# cut out slanted lines
d = 0.5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12, linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
# set labels
fig.supxlabel('Time (h)')
fig.supylabel('Rate of change (umol/h)')
fig.suptitle(fr'Initial $[{protac_id}] = {bpd_ec}\mu$M, $\alpha = {alpha}$, ' + r'$k_{ub}$' + fr'$= {k_ub}/h$')
# set legends
handles1, labels1 = ax1.get_legend_handles_labels()
labels1[1] = "Naked ternary"
labels1[2] = "Fully Ub'd ternary"
labels1[3] = "Fully Ub'd target"
ax1.legend(handles=handles1[1:], labels=labels1[1:], loc='upper right')
handles2, labels2 = ax2.get_legend_handles_labels()
labels2[0] = "Total target"
ax2.legend(handles=handles2[0:1], labels=labels2[0:1], loc='lower right')
# set legend text
plt.setp(ax1.get_legend().get_texts(), fontsize='15')
plt.setp(ax2.get_legend().get_texts(), fontsize='15')
# save plot
plt.savefig(f"./plots/{result_id}_rates.png")
