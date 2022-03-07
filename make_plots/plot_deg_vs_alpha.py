"""This script plots the simulation of degradation at different times across cooperativity values.
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
t_evals = [1, 6, 24]

result = [
    pd.read_csv(f"./saved_objects/{protac_id}_bpd_ec={bpd_ec}_t={t}_deg_vs_alpha.csv")
    for t in t_evals
    ]
result = pd.concat(result, ignore_index=True)

alpha_range = result['alpha'].unique()

result = result[['relative_target', 'relative_all_ternary', 'alpha', 't']]
result = result.rename(columns={'relative_target': 'Degradation', 'relative_all_ternary': 'Ternary'})
result = result.melt(id_vars=['alpha', 't'])

sns.set_style("whitegrid")
fig, ax = plt.subplots()

pal = sns.color_palette("Oranges", n_colors=9).as_hex()

p = sns.lineplot(
    data=result,
    x='alpha',
    y='value',
    hue='t',
    style='variable',
    palette=[pal[i] for i in [4, 6, 8]],
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
labels[0] = ""
for i in range(1, 4):
    t = float(labels[i])
    if t.is_integer():
        t = int(t)
    labels[i] = fr"$t = {t}$ h"
labels[4] = ""
ax.legend(handles=handles[1:4], labels=labels[1:4], loc='lower left', borderaxespad=0.25)
plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text
plt.savefig(f"plots/ACBI1_deg_vs_alpha.eps", bbox_inches='tight', dpi=1200)
