import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (10, 7)

protac_id = 'ACBI1'
t_all = [1, 6, 12, 24]
bpd_ec = 0.001

result_all = []
for i, t in enumerate(t_all):
    result_id = f"{protac_id}_bpd_ec={bpd_ec}_t={t}_kprod_vs_alpha_DEG"
    result = pd.read_csv(f"./saved_objects/{result_id}.csv")

    if i == 0:
        alpha_range = result['alpha'].unique()
        bpd_ec = result['initial_BPD_ec_conc'].unique()[0]

    result = result[['t', 'relative_target', 'relative_all_ternary', 'alpha', 'Conc_T_base']]
    result = result.astype({'t': 'int32'})
    result = result.rename(columns={'relative_target': 'Degradation', 'relative_all_ternary': 'Ternary'})
    result = result.melt(id_vars=['t', 'alpha', 'Conc_T_base'])

    result_all.append(result)

result_all = pd.concat(result_all)

sns.set_style("whitegrid")
g = sns.relplot(
    data=result_all,
    x='alpha',
    y='value',
    hue='Conc_T_base',
    style='variable',
    palette='Set2',
    kind='line',
    linewidth=2,
    col='t',
    col_wrap=2,
)

(
    g.set_axis_labels('', '')
     .set_titles('{col_var} = {col_name}h')
)

g.figure.subplots_adjust(wspace=0.1, hspace=0.1)
g.figure.supxlabel(r'Cooperativity ($\alpha$)')
g.figure.supylabel('% Baseline Target Protein')
# g.figure.suptitle(f'Target Protein Degradation with [{protac_id}]'
#                   + r'$_{ec}$'
#                   + fr'${bpd_ec}\mu$M')

plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(-5, 120)

ax = g.axes[0]
handles, labels = ax.get_legend_handles_labels()
labels[0] = "Initial [Target] (uM)"
for i in range(1, 5):
    conc = float(labels[i])
    if conc.is_integer():
        conc = int(conc)
    labels[i] = fr"[Target] $= {conc} \mu$M"
labels[5] = ""

for t, l in zip(g.legend.texts, labels):
    t.set_text(l)

g.legend.set_bbox_to_anchor((1.07, 0.6))
plt.setp(g.legend.get_texts(), fontsize='15')  # for legend text
# plt.setp(g.legend.get_title(), fontsize='15')  # for legend title

plt.savefig(f'plots/{protac_id}_bpd_ec={bpd_ec}_all_kprod_vs_alpha_DEG.png', bbox_inches='tight')
