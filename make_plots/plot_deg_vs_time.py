"""This script plots degradation and species amounts over time.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["figure.figsize"] = (3, 6)

protac_ids = ['PROTAC 1', 'ACBI1']
test_id = '&'.join(protac_ids)
test_id = test_id.replace(" ", "")
# test_id = 'ACBI1'
bpd_ec = 0.001
t = 24

result_id = f'{test_id}_bpd_ec={bpd_ec}_t={t}'
result = pd.read_csv(f"./saved_objects/{result_id}.csv")

result_deg = result[['t', 'PROTAC', 'relative_target']]
result_deg = result_deg.rename(columns={'relative_target': 'Degradation'})
result_deg = result_deg.melt(id_vars=['PROTAC', 't'])

result_species = result[['t', 'PROTAC', 'total_target', 'total_ternary', 'total_poly_ub_target']]
result_species = result_species.assign(percent_ternary=lambda df: df.total_ternary / df.total_target * 100,
                                       percent_poly_ub_target=lambda df: df.total_poly_ub_target / df.total_target * 100)
result_species = result_species.rename(columns={'percent_ternary': 'Ternary complex',
                                                'percent_poly_ub_target': 'Free poly-ubiquitinated target'})
result_species = result_species.melt(id_vars=['PROTAC', 't'], value_vars=['Ternary complex', 'Free poly-ubiquitinated target'])

result_rate = result[['t', 'PROTAC', 'degradation_rate']]
result_rate = result_rate.rename(columns={'degradation_rate': 'Degradation rate'})
result_rate = result_rate.melt(id_vars=['PROTAC', 't'])

result_protac = result[['t', 'PROTAC', 'total_bpd_ic', 'BPD_ec']]
result_protac = result_protac.assign(total_bpd=lambda df: df.total_bpd_ic * 5000 + df.BPD_ec)

sns.set_style("ticks")

"""Degradation over time."""

fig, (ax, ax2) = plt.subplots(2, 1, sharex='all')

# degradation vs. time
sns.lineplot(
    data=result_deg,
    x='t',
    y='value',
    hue='PROTAC',
    palette='Set2',
    linewidth=2,
    ax=ax
)

# y-axis settings
ax.set_ylabel('% Baseline Target Protein')
ax.set_ylim(0, 120)

# legend
ax.legend(loc="upper right", borderaxespad=0.25, fontsize='10')

# species vs. time
sns.lineplot(
    data=result_species,
    x='t',
    y='value',
    hue='PROTAC',
    style='variable',
    palette='Set2',
    ax=ax2,
    legend=False
)

# y-axis settings
ax2.set_ylabel('% Target Occupancy', labelpad=15)
ax2.set_ylim(bottom=0, top=6)
ax2.set_yticks(ticks=range(6))

# ticks
ax.tick_params(labelsize=12, direction='in')
ax2.tick_params(labelsize=12, direction='in')

legend_handles = [Line2D([0], [0], ls='-', label=r'$\sum_{i=0}^4 T_i\cdot P\cdot E3$', color='black'),
                  Line2D([0], [0], ls='--', label=r'$T_4 + T_4\cdot P$', color='black')]
ax2.legend(handles=legend_handles, title="", loc='upper right', borderaxespad=0.25,
           fontsize='10')

# x-axis settings
ax2.set_xlabel('Time (h)')
plt.xlim(result_deg['t'].min(), result_deg['t'].max())
plt.xticks(ticks=np.arange(start=0, stop=result_deg['t'].max() + 1, step=4, dtype=int))

plt.subplots_adjust(hspace=0)

plt.savefig(f"./plots/{result_id}.eps", bbox_inches="tight", dpi=1200)

"""Total PROTAC over time"""

# fig, ax = plt.subplots()
# sns.lineplot(
#     data=result_protac,
#     x='t',
#     y='total_bpd',
#     hue='PROTAC',
#     palette='Set2',
#     linewidth=2,
#     ax=ax
# )
# ax.set_xlabel('Time (h)')
# ax.set_xlim(result_protac['t'].min(), result_protac['t'].max())
# ax.set_xticks(ticks=np.arange(start=0, stop=result_protac['t'].max() + 1, step=4, dtype=int))
# ax.set_ylabel('Total PROTAC (umol)')
# ax.set_ylim(1.999e-7, 2.001e-7)
# ax.tick_params(labelsize=12, direction='in')
# ax.legend(loc="upper right", fontsize='8', title_fontsize='8', borderaxespad=0.25)
#
# plt.savefig(f"./plots/{result_id}_total_protac.png", bbox_inches="tight", dpi=1200)

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
