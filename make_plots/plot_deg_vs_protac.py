import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (10, 7)

"""
Degradation vs PROTAC concentration
"""
test_id = 'PROTAC1&ACBI1'
t_eval = 6
bpd_ec = 0.001

result = pd.read_csv(f"./saved_objects/{test_id}_t={t_eval}_DEG.csv")
result = result[['initial_BPD_ec_conc', 'percent_degradation', 'PROTAC']]
result = result.rename(columns={'percent_degradation': 'Degradation'})
result = result.melt(id_vars=['PROTAC', 'initial_BPD_ec_conc'])

sns.set_style("ticks")

fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='initial_BPD_ec_conc',
    y='value',
    hue='PROTAC',
    palette='Set2',
    ax=ax
)
p.tick_params(labelsize=15, direction='in')

plt.xscale('log')
plt.xlim(result['initial_BPD_ec_conc'].min(), result['initial_BPD_ec_conc'].max())
plt.xlabel(r'Initial $[PROTAC]_{ec}$ $(\mu M)$')
plt.ylim(-5, 120)
plt.ylabel('% Degradation')

plt.title(fr'Target Protein Degradation at $t = {t_eval}h$')
# handles, labels = ax.get_legend_handles_labels()
# labels[3] = "\nVariable"
# ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15')  # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"./plots/{test_id}_t={t_eval}_DEG.png", bbox_inches="tight")
