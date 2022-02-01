"""This script plots the results of an analysis of the effect of cooperativity and ubiquitination rate on degradation.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (4, 3.5)

protac_id = 'ACBI1'
t = 6
bpd_ec = 0.001
result_id = f"{protac_id}_bpd_ec={bpd_ec}_t={t}_kub_vs_alpha"

result = pd.read_csv(f"./saved_objects/{result_id}.csv")
result = result.loc[result['kub'] != 10000]
alpha_range = result['alpha'].unique()

# result['net_kub'] = result.kub - 180.

result = result[['relative_target', 'relative_all_ternary', 'alpha', 'kub']]
result = result.rename(columns={
    'relative_target': 'Degradation',
    'relative_all_ternary': 'Ternary'
})
result = result.melt(id_vars=['alpha', 'kub'])

pal = sns.color_palette('Set2')
pal_hex = pal.as_hex()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='alpha',
    y='value',
    hue='kub',
    style='variable',
    palette='Set2',
    linewidth=1,
    alpha=1,
    ax=ax
)
# p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(0, 120)
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('% Baseline Target Protein')
# plt.title(
#     f'Target Protein Degradation with [{protac_id}]'
#     + r'$_{ec} = $'
#     + fr'${bpd_ec}\mu$M at $t = {t}h$'
# )
# plt.text(10 ** -0.65, 70, r'$K_d = 0.1$', horizontalalignment='left', size=16, color=pal_hex[0])
# plt.text(10 ** 0.15, 70, r'$K_d = 1$', horizontalalignment='left', size=16, color=pal_hex[1])
# plt.text(10 ** 1.1, 70, r'$K_d = 10$', horizontalalignment='left', size=16, color=pal_hex[2])
# plt.text(10 ** 2.1, 70, r'$K_d = 100$', horizontalalignment='left', size=16, color=pal_hex[3])
handles, labels = ax.get_legend_handles_labels()
# labels[0] = "Ubiquitination rate (1/h)"
for i in range(1, 4):
    labels[i] = r"$k_{ub} = $" + fr"${labels[i]}/h$"
labels[4] = ''
ax.legend(handles=handles[1:], labels=labels[1:], loc='lower left', borderaxespad=0.25)
plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text
# plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"plots/{result_id}.png", bbox_inches='tight', dpi=1200)
