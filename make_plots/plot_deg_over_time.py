"""This script plots degradation and species amounts over time.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (10, 7)

protac_ids = ['PROTAC 1', 'ACBI1']
test_id = '&'.join(protac_ids)
test_id = test_id.replace(" ", "")
bpd_ec = 0.001

result_id = f'{test_id}_BPD_ec={bpd_ec}_DEG'
result = pd.read_csv(f"./saved_objects/{result_id}.csv")

result_deg = result[['t', 'PROTAC', 'degradation']]
result_deg = result_deg.rename(columns={'degradation': 'Degradation'})
result_deg = result_deg.melt(id_vars=['PROTAC', 't'])

result_ternary = result[['t', 'PROTAC', 'all_Ternary']]
result_ternary = result_ternary.rename(columns={'all_Ternary': 'Ternary complex'})
result_ternary = result_ternary.melt(id_vars=['PROTAC', 't'])

result_rate = result[['t', 'PROTAC', 'degradation_rate']]
result_rate = result_rate.rename(columns={'degradation_rate': 'Degradation rate'})
result_rate = result_rate.melt(id_vars=['PROTAC', 't'])

"""Degradation over time."""
sns.set_style("ticks")

fig, ax = plt.subplots()
sns.lineplot(
    data=result_deg,
    x='t',
    y='value',
    hue='PROTAC',
    palette='Set2',
    ax=ax
)
ax.set_xlabel('Time (h)')
ax.set_xlim(result_deg['t'].min(), result_deg['t'].max())
ax.set_ylabel('% Baseline Target Protein')
ax.set_ylim(-5, 120)
ax.tick_params(labelsize=15, direction='in')
ax.legend(loc="upper left")

ax2 = ax.twinx()
sns.lineplot(
    data=result_ternary,
    x='t',
    y='value',
    hue='PROTAC',
    linestyle='--',
    palette='Set2',
    ax=ax2,
    legend=False
)
ax2.set_ylabel('Ternary complex formation (umol)')
ax2.set_ylim(0, result_ternary['value'].max() * 2.5)
ax2.tick_params(labelsize=15, direction='in')

plt.title(r'Target Protein Degradation with initial $[PROTAC]_{ec} = 1$ nM', y=1.05)
plt.setp(ax.get_legend().get_texts(), fontsize='15')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"./plots/{result_id}.png", bbox_inches="tight")

"""Species totals over time."""
# result = pd.read_csv(f"./saved_objects/{result_id}.csv")
# result.columns
# result = result[['t', 'PROTAC', 'total_target', 'total_target_ub', 'total_bpd_ic', 'total_naked_ternary', 'total_ternary']]
# result = result.melt(id_vars=['t', 'PROTAC'])
#
# sns.set_style("whitegrid")
# fig, (ax, ax2, ax3) = plt.subplots(3, 1, sharex=True)  # break y-axis into two portions
# fig.subplots_adjust(hspace=0.15)  # adjust space between axes
# # plot same data on both axes
# p = sns.lineplot(
#     data=result,
#     x='t',
#     y='value',
#     hue='variable',
#     style='PROTAC',
#     palette='Set2',
#     ax=ax
# )
# q = sns.lineplot(
#     data=result,
#     x='t',
#     y='value',
#     hue='variable',
#     style='PROTAC',
#     palette='Set2',
#     ax=ax2,
#     legend=False
# )
# r = sns.lineplot(
#     data=result,
#     x='t',
#     y='value',
#     hue='variable',
#     style='PROTAC',
#     palette='Set2',
#     ax=ax3,
#     legend=False
# )
# r.tick_params(labelsize=15)
# # limit the view to different portions of data
# ax.set_ylim(0.125e-13, 4e-13)  # total target
# ax2.set_ylim(0, 0.125e-13)  # total bpd_ic
# ax3.set_ylim(0, 1e-15)
# ax3.set_xlim(result['t'].min(), result['t'].max())
# # hide ylabels
# ax.set(ylabel=None, xlabel=None)
# ax2.set(ylabel=None, xlabel=None)
# ax3.set(ylabel=None, xlabel=None)
# # hide the spines between ax and ax2
# ax.spines.bottom.set_visible(False)
# ax2.spines.top.set_visible(False)
# ax2.spines.bottom.set_visible(False)
# ax.xaxis.tick_top()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# ax3.xaxis.tick_bottom()
# plt.xticks(np.arange(start=0, stop=24 + 1, step=4))
# # cut-out slanted lines
# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
# ax2.plot([0, 1, 0, 1], [0, 0, 1, 1], transform=ax2.transAxes, **kwargs)
# ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
#
# fig.supxlabel('Time (h)')
# fig.supylabel('Amount (umol)')
# fig.suptitle(r'Target Protein Degradation with initial $[PROTAC]_{ec} = 0.0005 \mu$M')
#
# handles, labels = ax.get_legend_handles_labels()
# labels[1] = "total Target"
# labels[2] = "total Ubiquitinated Target"
# labels[3] = "total intracellular PROTAC"
# labels[4] = "total naked Ternary complex"
# labels[5] = "total Ternary complex"
#
# labels[6] = "\n"
# ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# plt.setp(ax.get_legend().get_texts(), fontsize='15')  # for legend text
# plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
# plt.savefig(f"./plots/{result_id}_ic_amounts.png", bbox_inches="tight")
#
