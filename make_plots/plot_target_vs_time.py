"""This script plots degradation and species amounts over time.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Patch

# plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12
plt.rcParams["figure.figsize"] = (3, 3)

protac_ids = ['PROTAC 1', 'ACBI1']
test_id = '&'.join(protac_ids)
test_id = test_id.replace(" ", "")
# test_id = 'ACBI1'
bpd_ec = 0.001
t = 24

result_id = f'{test_id}_bpd_ec={bpd_ec}_t={t}'
result = pd.read_csv(f"./saved_objects/{result_id}.csv")

this_protac = 'PROTAC 1'

result = result.loc[result['PROTAC'] == this_protac]

target_df: pd.DataFrame = result.assign(
    t=result['t'],
    target=result.filter(regex='(^T$)|(^.*(?!T_Ub_4)(T_Ub.*))').sum(axis=1) / result['total_target'] * 100,
    poly_target=result.filter(regex='(^.*T_Ub_4$)').sum(axis=1) / result['total_target'] * 100,
    ternary=result.filter(regex='^Ternary.*').sum(axis=1) / result['total_target'] * 100
)
target_df = target_df[['t',
                       'target',
                       'poly_target',
                       'ternary']]

labels = ['Target',
          'Poly-ub target',
          'Ternary']

cmap = cm.get_cmap('Paired')

fig, (ax, ax2) = plt.subplots(2, 1, sharex='all')  # break y-axis into two portions
fig.subplots_adjust(hspace=0.15)  # adjust space btwn axes

ax.stackplot(
    target_df.t.tolist(),
    target_df.target,
    target_df.poly_target,
    target_df.ternary,
    # labels=labels,
    colors=cmap.colors[:target_df.shape[1]]
)
ax2.stackplot(
    target_df.t.tolist(),
    target_df.target,
    target_df.poly_target,
    target_df.ternary,
    labels=labels,
    colors=cmap.colors[:target_df.shape[1]]
)

ax.set_xlim(0, 24)
ax.set_ylim(95, 100)
ax2.set_xlim(0, 24)
ax2.set_ylim(0, 95)

# hide the spines between ax and ax2
ax.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# ticks
plt.setp(ax.get_yticklabels(), fontsize=10)
plt.setp(ax2.get_yticklabels(), fontsize=10)
plt.xticks(ticks=np.arange(0, 25, 4), fontsize=10)

# cut-out slanted lines
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

fig.supxlabel('Time (h)', y=-0.05, fontsize='12')
fig.supylabel(r"% Target Occupancy", x=-0.05, fontsize='12')

# legend
handles, labels = ax2.get_legend_handles_labels()
labels[0] = r'$\sum_{i=0}^3 T_i + T_i\cdot P$'
labels[1] = r'$T_4 + T_4\cdot P$'
labels[2] = r'$\sum_{i=0}^4 T_i\cdot P\cdot E3$'
ax2.legend(handles=handles, labels=labels, loc='lower right', fontsize='8')
# ax2.legend(handles=legend_handles, title="", loc='lower right', borderaxespad=0.25,
#            fontsize='8')

# plt.setp(ax2.get_legend().get_texts(), fontsize='8')
plt.savefig(f"./plots/{this_protac.replace(' ', '')}_bpd_ec={bpd_ec}_t={t}_target_occupancy.eps", dpi=1200, bbox_inches="tight")
