
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
result_id = f"{protac_id}t={t}_ps_cell_vs_alpha_DEG"

result = pd.read_csv(f"./saved_objects/{result_id}.csv")
alpha_range = result['alpha'].unique()
bpd_ec = result['initial_BPD_ec_conc'].unique()[0]


result = result[['alpha', 'degradation', 'all_Ternary', 'PS_cell']]
result = result.rename(columns={
    'degradation': 'Degradation',
    'all_Ternary': 'Ternary'
})
result_melt = result.melt(id_vars=['alpha', 'PS_cell'])

pal = sns.color_palette('Set2')
pal_hex = pal.as_hex()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result_melt,
    x='alpha',
    y='value',
    hue='PS_cell',
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
handles, labels = ax.get_legend_handles_labels()
labels[0] = "Permeability surface area (L/h)"
ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.savefig(f"plots/{result_id}.png", bbox_inches='tight')
