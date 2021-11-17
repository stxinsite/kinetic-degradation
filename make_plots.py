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
plt.rcParams["figure.figsize"] = (12,8)

PROTAC_ID = 'SiTX_38404'
result = pd.read_csv(f"saved_objects/{PROTAC_ID}_kon_vs_alpha.csv")
alpha_range = result['alpha'].unique()
result = result[['Target_deg', 'relative_Ternary', 'relative_all_Ternary', 'alpha', 'kon_T_binary']]
result_melt = result.melt(id_vars = ['alpha', 'kon_T_binary'])
result_melt['PROTAC'] = '38404'

result_38406 = pd.read_csv(f"saved_objects/SiTX_38406_kon_vs_alpha.csv")
result_38406 = result_38406[['Target_deg', 'relative_Ternary', 'relative_all_Ternary', 'alpha', 'kon_T_binary']]
result_38406_melt = result_38406.melt(id_vars = ['alpha', 'kon_T_binary'])
result_38406_melt['PROTAC'] = '38406'

p = sns.lineplot(
    data = pd.concat([result_melt, result_38406_melt], ignore_index=True), # result_melt,
    x = 'alpha',
    y = 'value',
    hue = 'PROTAC', # 'kon_T_binary',
    style = 'variable',
    palette = 'Set2'
)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.ylim(0, 120)
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel('% Baseline Target Protein')
plt.title(r'Varying $\alpha$ with initial $[BPD]_{ec}$ = 0.1uM at 24h')
plt.savefig("plots/SiTX_38404&38406_kon_vs_alpha.png")

plt.savefig(f'plots/{PROTAC_ID}_kon_vs_alpha.png')


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
