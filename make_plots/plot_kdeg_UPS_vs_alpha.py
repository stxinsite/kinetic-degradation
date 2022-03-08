"""This script plots the simulation of degradation at different times across kdeg values.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["figure.figsize"] = (3, 3)

protac_id = 'ACBI1'
bpd_ec = 0.001
t = 6

result_id = f"{protac_id}_bpd_ec={bpd_ec}_t={t}_deg_vs_kdeg"
result = pd.read_csv(
    f"./saved_objects/{result_id}.csv")

alpha_range = result['alpha'].unique()

result = result[['relative_target',
                 'relative_all_ternary', 'alpha', 'kdeg_UPS']]
result = result.rename(
    columns={'relative_target': 'Degradation',
             'relative_all_ternary': 'Ternary'}
)
result = result.melt(id_vars=['alpha', 'kdeg_UPS'])

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='alpha',
    y='value',
    hue='kdeg_UPS',
    style='variable',
    palette='PuOr',
    linewidth=1.25,
    ax=ax
)
# p.tick_params(labelsize=12)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(0, 120)
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('% Baseline Target Protein')

handles, labels = ax.get_legend_handles_labels()
for i in range(1, 4):
    kdeg = float(labels[i])
    if kdeg.is_integer():
        kdeg = int(kdeg)
    labels[i] = r"$k_{deg,T} = $" + f"${kdeg}/h$"
labels[4] = ""
ax.legend(handles=handles[1:4], labels=labels[1:4],
          loc='upper right', borderaxespad=0.25)
plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text
plt.savefig(f"plots/{result_id}.eps", bbox_inches='tight', dpi=1200)
