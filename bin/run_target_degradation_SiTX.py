import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import kinetic_module.kinetic_tests as kinetic_tests
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,8)

"""
LOAD PARAMETERS FROM CONFIG
"""
with open('./data/SiTX_38404_config.yml', 'r') as file:
    params_SiTX_38404 = yaml.safe_load(file)

with open('./data/SiTX_38406_config.yml', 'r') as file:
    params_SiTX_38406 = yaml.safe_load(file)

"""
RUN TEST(S)
"""
t = [24]  # time points at which to evaluate solver
initial_BPD_ec_conc = np.logspace(base = 10.0, start = -1, stop = 5, num = 50) / 1000  # various initial concentrations of BPD_ec (uM)

degradation_df_38404 = kinetic_tests.solve_target_degradation(params_SiTX_38404, t, initial_BPD_ec_conc, 'SiTX_38404')
degradation_df_38406 = kinetic_tests.solve_target_degradation(params_SiTX_38406, t, initial_BPD_ec_conc, 'SiTX_38406')

degradation_df_38404['PROTAC'] = '38404'
degradation_df_38406['PROTAC'] = '38406'

frames = [degradation_df_38404, degradation_df_38406]
result = pd.concat(frames)
result.to_csv("saved_objects/SiTX_PROTAC_result.csv")

# p = sns.lineplot(data = result, x = 'Conc_BPD_ec', y = 'Target_deg', hue = 'PROTAC', palette = "Set2")
# plt.xscale('log')
# plt.xlim(initial_BPD_ec_conc.min(), initial_BPD_ec_conc.max())
# plt.ylim(0, 120)
# plt.xlabel('BPD Concentration (uM)')
# plt.ylabel('% Baseline Target Protein')
# plt.title(f'Percent baseline Target protein at {str(t[0])} hours')
# plt.savefig(f'plots/Target_Deg_t={str(t[0])}h.png')
