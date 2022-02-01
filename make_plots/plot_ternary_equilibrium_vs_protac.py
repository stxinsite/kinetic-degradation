import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kinetic_module.equilibrium_functions import predict_ternary

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["figure.figsize"] = (4, 3.5)
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
result = result[['PROTAC', 'initial_BPD_ec_conc', 'total_target', 'total_bpd_ic', 'total_ternary']]
# convert amounts to concentrations
result = result.assign(total_ternary=lambda df: df.total_ternary / v_ic,
                       total_bpd_ic=lambda df: df.total_bpd_ic / v_ic,
                       total_target=lambda df: df.total_target / v_ic)
result['ternary_equilibrium'] = -1.0
result['ternary_equilibrium_static'] = -1.0

for idx, dat in result.iterrows():
    total_target = dat['total_target']
    total_protac = dat['total_bpd_ic']
    total_e3 = 0.1
    if dat['PROTAC'] == 'PROTAC 1':
        kd_target = 8.54
        kd_e3 = 1.23e-2
        alpha = 3.2
    elif dat['PROTAC'] == 'ACBI1':
        kd_target = 9.26
        kd_e3 = 0.0694
        alpha = 26
    else:
        raise ValueError('PROTAC not available.')

    ternary = predict_ternary(total_target, total_protac, total_e3, kd_target, kd_e3, alpha)
    ternary_static = predict_ternary(1.0, total_protac, 0.1, kd_target, kd_e3, alpha)

    result.at[idx, 'ternary_equilibrium'] = ternary
    result.at[idx, 'ternary_equilibrium_static'] = ternary_static

result = result.rename(columns={'total_ternary': 'Ternary complex',
                                'ternary_equilibrium': 'Equilibrium solution'})

# ternary formation data
result_ternary = result.melt(id_vars=['PROTAC', 'initial_BPD_ec_conc'],
                             value_vars=['Ternary complex', 'Equilibrium solution'])

sns.set_style("ticks")

# fig, ax = plt.subplots()
#
# # degradation curves
# p = sns.lineplot(
#     data=result_ternary,
#     x='initial_BPD_ec_conc',
#     y='value',
#     hue='PROTAC',
#     palette='Set2',
#     style='variable',
#     linewidth=2,
#     ax=ax
# )
#
# # legend
# handles, labels = ax.get_legend_handles_labels()
# labels[3] = ""
# ax.legend(handles=handles[1:], labels=labels[1:], title="", loc='upper right', borderaxespad=0.25)
# plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text
#
# # x-axis settings
# ax.set_xlabel(r'Concentration ($\mu$M)')
#
# # y-axis settings
# ax.set_ylim(bottom=0)
# ax.set_ylabel('Ternary Complex Formation (uM)')
#
# # figure-level x-axis settings
# plt.xscale('log')
# plt.xlim(result['initial_BPD_ec_conc'].min(), result['initial_BPD_ec_conc'].max())
# plt.xticks(np.power(10, np.arange(-4, 3, dtype=float)))
#
# plt.savefig(f"./plots/{result_id}_equilibrium.png", dpi=1200, bbox_inches="tight")

# STATIC TERNARY COMPLEX EQUILIBRIUM
fig, ax = plt.subplots()

# degradation curves
q = sns.lineplot(
    data=result,
    x='initial_BPD_ec_conc',
    y='ternary_equilibrium_static',
    hue='PROTAC',
    palette='Set2',
    linewidth=2,
    ax=ax
)

# legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles, labels=labels, title="", loc='upper left', borderaxespad=0.25)
plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text

# x-axis settings
ax.set_xlabel(r'Concentration ($\mu$M)')

# y-axis settings
ax.set_ylim(bottom=0)
ax.set_ylabel('Ternary Complex Formation (uM)')

# figure-level x-axis settings
plt.xscale('log')
plt.xlim(result['initial_BPD_ec_conc'].min(), result['initial_BPD_ec_conc'].max())
plt.xticks(np.power(10, np.arange(-4, 3, dtype=float)))

plt.savefig(f"./plots/{result_id}_equilibrium_static.png", dpi=1200, bbox_inches="tight")
