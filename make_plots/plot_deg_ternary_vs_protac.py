import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["figure.figsize"] = (4, 7)
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["ytick.labelsize"] = 12

"""
Degradation vs PROTAC concentration
"""
test_id = 'PROTAC1&ACBI1'
t_eval = 6
bpd_ec = 0.001
v_ic = 5.24e-13
result_id = f"{test_id}_target=1_e3=0.1_t={t_eval}"

result = pd.read_csv(f"./saved_objects/{result_id}.csv")
result = result[['PROTAC', 'initial_BPD_ec_conc', 'relative_target', 'total_ternary', 'total_bpd_ic']]
# convert amounts to concentrations
result = result.assign(total_ternary=lambda df: df.total_ternary / v_ic,
                       total_bpd_ic=lambda df: df.total_bpd_ic / v_ic)
result = result.rename(columns={'relative_target': 'Degradation',
                                'total_ternary': 'Ternary complex',
                                'total_bpd_ic': 'Intracellular PROTAC'})

# degradation data
result_deg = result.melt(id_vars=['PROTAC', 'initial_BPD_ec_conc'], value_vars=['Degradation'])
# ternary formation data
result_ternary = result.melt(id_vars=['PROTAC', 'initial_BPD_ec_conc'], value_vars=['Ternary complex'])
# intracellular PROTAC data
result_protac = result.melt(id_vars=['PROTAC', 'initial_BPD_ec_conc'], value_vars=['Intracellular PROTAC'])

sns.set_style("ticks")

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')

# degradation curves
p = sns.lineplot(
    data=result_deg,
    x='initial_BPD_ec_conc',
    y='value',
    hue='PROTAC',
    palette='Set2',
    linewidth=2,
    ax=ax1
)

# y-axis settings
ax1.set_ylim(bottom=0, top=120)
ax1.set_ylabel('% Baseline Target Protein')

# legend
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=labels, title="", loc='upper right', borderaxespad=0.25)
plt.setp(ax1.get_legend().get_texts(), fontsize='8')  # for legend text

# ternary complex formation curves
q = sns.lineplot(
    data=result_ternary,
    x='initial_BPD_ec_conc',
    y='value',
    hue='PROTAC',
    palette='Set2',
    linewidth=2,
    ax=ax2,
    legend=False
)

# x-axis settings
ax2.set_xlabel(r'Concentration ($\mu$M)')

# y-axis settings
ax2.set_ylim(bottom=0)
ax2.set_ylabel('Ternary Complex (uM)')

# create another Axes with shared x-axis
ax3 = ax2.twinx()

r = sns.lineplot(
    data=result_protac,
    x='initial_BPD_ec_conc',
    y='value',
    hue='PROTAC',
    palette='Set2',
    linewidth=2,
    ax=ax3,
    linestyle='--',
    legend=False
)

# y-axis settings
ax3.set_ylim(bottom=result_protac['value'].min(), top=result_protac['value'].max())
ax3.set_yscale('log')
ax3.set_ylabel('Total Intracellular PROTAC (uM)')

# legend
legend_handles = [Line2D([0], [0], ls='-', label='Ternary complex', color='black'),
                  Line2D([0], [0], ls='--', label='Total PROTAC', color='black')]
ax3.legend(handles=legend_handles, title="", loc='upper right', borderaxespad=0.25)
plt.setp(ax3.get_legend().get_texts(), fontsize='8')  # for legend text

# figure-level x-axis settings
plt.xscale('log')
plt.xlim(result['initial_BPD_ec_conc'].min(), result['initial_BPD_ec_conc'].max())
plt.xticks(np.power(10, np.arange(-4, 3, dtype=float)))

# fig.tight_layout()
plt.subplots_adjust(hspace=0)

plt.savefig(f"./plots/{result_id}.png", dpi=1200, bbox_inches="tight")
