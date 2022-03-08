import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (3, 3)

protac_id = 'ACBI1'
t = 6
bpd_ec = 0.001
result_id = f"{protac_id}_bpd_ec={bpd_ec}_t={t}_kprod_vs_alpha"

result = pd.read_csv(f"./saved_objects/{result_id}.csv")
alpha_range = result['alpha'].unique()
bpd_ec = result['initial_BPD_ec_conc'].unique()[0]

result = result[['relative_target', 'relative_all_ternary', 'alpha', 'Conc_T_base']]
result = result.rename(columns={'relative_target': 'Degradation', 'relative_all_ternary': 'Ternary'})
result = result.melt(id_vars=['alpha', 'Conc_T_base'])

pal = sns.color_palette('Set2')
pal_hex = pal.as_hex()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='alpha',
    y='value',
    hue='Conc_T_base',
    style='variable',
    palette='Dark2',
    linewidth=1.25,
    ax=ax
)
# p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(0, 120)
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('% Baseline Target Protein')
# plt.title(
#     f"Target Protein Degradation with [{protac_id}]"
#     + r"$_{ec} = $"
#     + fr"${bpd_ec}\mu$M at $t = {int(t)}h$"
# )
handles, labels = ax.get_legend_handles_labels()
# labels[0] = "Initial [Target]"
for i in range(1,5):
    target_conc = float(labels[i])
    if target_conc.is_integer():
        target_conc = int(target_conc)
    labels[i] = fr"$[T] = {str(target_conc)}$" + r" $\mu$M"
labels[5] = ""
ax.legend(handles=handles[1:5], labels=labels[1:5], loc='lower left', borderaxespad=0.25)
plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text
# plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"plots/{result_id}.eps", bbox_inches="tight", dpi=1200)
