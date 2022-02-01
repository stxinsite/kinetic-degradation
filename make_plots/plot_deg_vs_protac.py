import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["figure.figsize"] = (4, 3.5)

"""
Degradation vs PROTAC concentration
"""
test_id = 'PROTAC1&ACBI1'
t_eval = 6
bpd_ec = 0.001
result_id = f"{test_id}_target=1_e3=0.1_t={t_eval}"

result = pd.read_csv(f"./saved_objects/{result_id}.csv")
result = result[['initial_BPD_ec_conc', 'relative_target', 'PROTAC']]
result = result.rename(columns={'relative_target': 'Degradation'})
result = result.melt(id_vars=['PROTAC', 'initial_BPD_ec_conc'])

sns.set_style("ticks")

fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='initial_BPD_ec_conc',
    y='value',
    hue='PROTAC',
    palette='Set2',
    linewidth=2,
    ax=ax
)
p.tick_params(labelsize=12, direction='in')

plt.xscale('log')
plt.xlim(result['initial_BPD_ec_conc'].min(), result['initial_BPD_ec_conc'].max())
plt.xlabel(r'Concentration ($\mu$M)')
plt.ylim(0, 120)
plt.ylabel('% Baseline Target Protein')

# plt.title(fr'Target Protein Degradation at $t = {t_eval}h$')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title="", loc='upper right', borderaxespad=0.25)
plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text
# plt.setp(ax.get_legend().get_title(), fontsize='15')  # for legend title
plt.savefig(f"./plots/{result_id}.png", dpi=1200, bbox_inches="tight")
