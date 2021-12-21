"""This script plots the results of an analysis of the effect of cooperativity
and ternary complex degradation rate on degradation.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (10,7)

protac_id = 'ACBI1'
t = 6
result_id = f"{protac_id}t={t}_kdeg_ternary_vs_alpha_DEG"

result = pd.read_csv(f"./saved_objects/{result_id}.csv")
alpha_range = result['alpha'].unique()
bpd_ec = result['initial_BPD_ec_conc'].unique()[0]

"""Degradation curves."""
result_deg = result[['degradation', 'all_Ternary', 'alpha', 'kdeg_Ternary']]
result_deg = result_deg.rename(columns={
    'degradation': 'Degradation',
    'all_Ternary': 'Ternary'
})
result_deg = result_deg.melt(id_vars=['alpha', 'kdeg_Ternary'])

pal = sns.color_palette('Set2')
pal_hex = pal.as_hex()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result_deg,
    x='alpha',
    y='value',
    hue='kdeg_Ternary',
    style='variable',
    palette='Set2',
    linewidth=2,
    alpha=1,
    ax=ax
)
p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(-5, 120)
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('% Baseline Target Protein')
plt.title(
    f'Target Protein Degradation with [{protac_id}]'
    + r'$_{ec} = $'
    + fr'${bpd_ec}\mu$M at $t = {t}h$',
    y=1.05
)
# plt.text(10 ** -0.65, 70, r'$K_d = 0.1$', horizontalalignment='left', size=16, color=pal_hex[0])
# plt.text(10 ** 0.15, 70, r'$K_d = 1$', horizontalalignment='left', size=16, color=pal_hex[1])
# plt.text(10 ** 1.1, 70, r'$K_d = 10$', horizontalalignment='left', size=16, color=pal_hex[2])
# plt.text(10 ** 2.1, 70, r'$K_d = 100$', horizontalalignment='left', size=16, color=pal_hex[3])
handles, labels = ax.get_legend_handles_labels()
# labels[0] = "Ubiquitination rate (1/h)"
labels[0] = 'Ternary Degradation Rate (1/h)'
ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"plots/{result_id}.png", bbox_inches='tight')


result_rates = result[['kdeg_Ternary', 'alpha', 'degradation_rate']]
result_rates = result_rates.melt(id_vars=['kdeg_Ternary', 'alpha'])

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result_rates,
    x='alpha',
    y='value',
    hue='kdeg_Ternary',
    style='variable',
    palette='Set2',
    linewidth=2,
    alpha=1,
    ax=ax
)
p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('umol/h')
plt.title(
    f'Total Target Rate of Change with [{protac_id}]'
    + r'$_{ec} = $'
    + fr'${bpd_ec}\mu$M at $t = {t}h$',
    y=1.05
)
# plt.text(10 ** -0.65, 70, r'$K_d = 0.1$', horizontalalignment='left', size=16, color=pal_hex[0])
# plt.text(10 ** 0.15, 70, r'$K_d = 1$', horizontalalignment='left', size=16, color=pal_hex[1])
# plt.text(10 ** 1.1, 70, r'$K_d = 10$', horizontalalignment='left', size=16, color=pal_hex[2])
# plt.text(10 ** 2.1, 70, r'$K_d = 100$', horizontalalignment='left', size=16, color=pal_hex[3])
handles, labels = ax.get_legend_handles_labels()
# labels[0] = "Ubiquitination rate (1/h)"
labels[0] = 'Ternary Degradation Rate (1/h)'
ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"plots/{result_id}_rate.png", bbox_inches='tight')
