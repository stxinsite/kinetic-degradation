import numpy as np
import pandas as pd
import seaborn as sns
from sigfig import round
import matplotlib.pyplot as plt
import json
import multiprocessing
from src.kinetic_module.calc_full_config import KineticParameters
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (10,7)

"""
ktransit_UPS and alpha
"""
result = pd.read_csv("./saved_objects/SiTX_38404_ktransit_vs_alpha_DEG.csv")
alpha_range = result['alpha'].unique()
result.shape
result = result[['degradation', 'all_Ternary', 'Dmax', 'alpha', 'ktransit_UPS']]
# result = result[['degradation', 'all_Ternary', 'alpha', 'ktransit_UPS']]
result = result.rename(columns = {'degradation': 'Degradation', 'all_Ternary': 'Ternary'})
result = result.melt(id_vars = ['alpha', 'ktransit_UPS'])

pal = sns.color_palette('Set2')
pal_hex = pal.as_hex()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data = result,
    x = 'alpha',
    y = 'value',
    hue = 'ktransit_UPS',
    style = 'variable',
    palette = 'Set2',
    linewidth = 3,
    ax = ax
)
p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(-5, 120)
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('% Baseline Target Protein')
plt.title(r'Target Protein Degradation with $[{PROTAC 1}]_{ec} = 0.1\mu$M at $t = 6h$')
# plt.text(10 ** -0.65, 70, r'$K_d = 0.1$', horizontalalignment='left', size=16, color=pal_hex[0])
# plt.text(10 ** 0.15, 70, r'$K_d = 1$', horizontalalignment='left', size=16, color=pal_hex[1])
# plt.text(10 ** 1.1, 70, r'$K_d = 10$', horizontalalignment='left', size=16, color=pal_hex[2])
# plt.text(10 ** 2.1, 70, r'$K_d = 100$', horizontalalignment='left', size=16, color=pal_hex[3])
handles, labels = ax.get_legend_handles_labels()
labels[0] = r"$k_{transit}$"
# labels[5] = "Variable"
ax.legend(handles=handles, labels=labels, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.savefig(f"plots/SiTX_38404_ktransit_vs_alpha_DEG.png", bbox_inches='tight')


"""
Degradation and Dmax vs PROTAC concentration
"""
result = pd.read_csv("./saved_objects/SiTX_38404+38406_t=6h_DEG.csv")
result.head()
result = result[['initial_BPD_ec_conc', 'degradation', 'Dmax', 'PROTAC']]
result = result.rename(columns = {'degradation': 'Degradation'})
result = result.melt(id_vars = ['PROTAC', 'initial_BPD_ec_conc'])
result.head()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data = result,
    x = 'initial_BPD_ec_conc',
    y = 'value',
    hue = 'PROTAC',
    style = 'variable',
    palette = 'Set2',
    ax = ax
)
p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(result['initial_BPD_ec_conc'].min(), result['initial_BPD_ec_conc'].max())
plt.ylim(-5, 120)
plt.xlabel(r'Initial $[PROTAC]_{ec}$ $(\mu M)$')
plt.ylabel('% Baseline Target Protein')
plt.title(r'Target Protein Degradation and Dmax at $t = 6h$')
handles, labels = ax.get_legend_handles_labels()
labels[3] = "\nVariable"
ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.savefig(f"./plots/SiTX_38404+38406_t=6h_DEG.png", bbox_inches = "tight")


"""
Degradation over time
"""
result = pd.read_csv("./saved_objects/SiTX_38404+38406_DEG.csv")
result = result[['t', 'degradation', 'all_Ternary', 'PROTAC']]
result = result.rename(columns = {'degradation': 'Degradation', 'all_Ternary': 'Ternary complex'})
result = result.melt(id_vars = ['PROTAC', 't'])
# result = result.rename(columns = {'variable': 'Measure'})

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data = result,
    x = 't',
    y = 'value',
    hue = 'PROTAC',
    style = 'variable',
    palette = 'Set2',
    ax = ax
)
p.tick_params(labelsize=15)
plt.xlim(result['t'].min(), result['t'].max())
plt.ylim(-5, 120)
plt.xlabel('Time (h)')
plt.ylabel('% Baseline Target Protein')
plt.title(r'Target Protein Degradation with initial $[PROTAC]_{ec} = 0.1 \mu$M')
handles, labels = ax.get_legend_handles_labels()
labels[3] = "\nVariable"
ax.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.savefig(f"./plots/SiTX_38404+38406_DEG.png", bbox_inches = "tight")

"""
Permeability
"""
PROTAC_ID = 'SiTX_38406'

result = pd.read_csv(f"saved_objects/{PROTAC_ID}_Permeability_DEG.csv")
PS_cell_range = result['PS_cell'].unique()
result.head()
result.shape
result = result[['t', 'degradation', 'Ternary', 'all_Ternary', 'PS_cell']]
result_melt = result.melt(id_vars = ['PS_cell', 't'])
result_melt.head()

sns.set_style("whitegrid")
p = sns.lineplot(
    data = result_melt,
    x = 't',
    y = 'value',
    hue = 'PS_cell',
    style = 'variable',
    palette = 'Set2'
)
plt.xlim(result['t'].min(), result['t'].max())
plt.ylim(-5, 120)
plt.xlabel('Time (h)')
plt.ylabel('% Baseline Target Protein')
plt.title(f'Degradation with initial [{PROTAC_ID}] = 0.1uM')
plt.savefig(f"plots/{PROTAC_ID}_Permeability_DEG.png")


"""
Kd and alpha
"""
result = pd.read_csv(f"./saved_objects/SiTX_38406_Kd_vs_alpha_DEG.csv")
alpha_range = result['alpha'].unique()
result.shape
result['t'].unique()
result = result[['degradation', 'all_Ternary', 'alpha', 'Kd_T_binary']]
result = result.rename(columns = {'degradation': 'Degradation', 'all_Ternary': 'Ternary'})
result = result.melt(id_vars = ['alpha', 'Kd_T_binary'])
# result = result.rename(columns = {'variable': 'Measure'})

pal = sns.color_palette('Set2')
pal_hex = pal.as_hex()

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data = result,
    x = 'alpha',
    y = 'value',
    hue = 'Kd_T_binary',
    style = 'variable',
    palette = 'Set2',
    linewidth = 3,
    ax = ax
)
p.tick_params(labelsize=15)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(-5, 120)
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('% Baseline Target Protein')
plt.title(r'Target Protein Degradation with $[ACBI1]_{ec} = 0.1\mu$M at $t = 6h$')
plt.text(10 ** -0.65, 70, r'$K_d = 0.1$', horizontalalignment='left', size=16, color=pal_hex[0])
plt.text(10 ** 0.15, 70, r'$K_d = 1$', horizontalalignment='left', size=16, color=pal_hex[1])
plt.text(10 ** 1.1, 70, r'$K_d = 10$', horizontalalignment='left', size=16, color=pal_hex[2])
plt.text(10 ** 2.1, 70, r'$K_d = 100$', horizontalalignment='left', size=16, color=pal_hex[3])
handles, labels = ax.get_legend_handles_labels()
labels[5] = "Variable"
ax.legend(handles=handles[5:], labels=labels[5:], bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.setp(ax.get_legend().get_texts(), fontsize='15') # for legend text
plt.setp(ax.get_legend().get_title(), fontsize='15') # for legend title
plt.savefig(f"plots/SiTX_38406_Kd_vs_alpha_DEG.png", bbox_inches = "tight")




t = [24]  # time points at which to evaluate solver
initial_BPD_ec_conc = np.logspace(base = 10.0, start = -1, stop = 5, num = 50) / 1000  # various initial concentrations of BPD_ec (uM)

result = pd.read_csv("saved_objects/SiTX_PROTAC_result.csv")
result.head()

p = sns.lineplot(
    data = result,
    x = 'Conc_BPD_ec',
    y = 'Target_deg',
    hue = 'PROTAC',
    palette = "Set2"
)
plt.xscale('log')
plt.xlim(initial_BPD_ec_conc.min(), initial_BPD_ec_conc.max())
plt.ylim(0, 120)
plt.xlabel('BPD Concentration (uM)')
plt.ylabel('% Baseline Target Protein')
plt.title(f'Percent baseline Target protein at {str(t[0])} hours')
plt.savefig(f'plots/Target_Deg_t={str(t[0])}h.png')

t = np.arange(start = 0, stop = 168 + 1, step = 3)  # time points at which to evaluate solver
initial_BPD_ec_conc = [0.1]  # initial concentration of BPD_ec (uM)
result = pd.read_csv("saved_objects/SiTX_PROTAC_fix_BPD_result.csv")

p = sns.lineplot(data = result, x = 't', y = 'Target_deg', hue = 'PROTAC', palette = "Set2")
plt.xlim(t.min(), t.max())
plt.ylim(0, 120)
plt.xlabel('Time (h)')
plt.ylabel('% Baseline Target Protein')
plt.title('Percent baseline Target protein at [$BPD_{ec}$] = ' + f'{str(initial_BPD_ec_conc)} uM')
plt.savefig(f'plots/Target_Deg_BPD={str(initial_BPD_ec_conc[0])}uM.png')
