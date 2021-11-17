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

t = np.arange(start = 0, stop = 168 + 1, step = 3)  # time points at which to evaluate solver
initial_BPD_ec_conc = [0.1]  # initial concentration of BPD_ec (uM)

degradation_df_38404 = kinetic_tests.solve_target_degradation(params_SiTX_38404, t, initial_BPD_ec_conc, 'SiTX_38404')
degradation_df_38406 = kinetic_tests.solve_target_degradation(params_SiTX_38406, t, initial_BPD_ec_conc, 'SiTX_38406')

degradation_df_38404['PROTAC'] = '38404'
degradation_df_38406['PROTAC'] = '38406'

frames = [degradation_df_38404, degradation_df_38406]
result = pd.concat(frames)
result.to_csv("saved_objects/SiTX_PROTAC_fix_BPD_result.csv")

# p = sns.lineplot(data = result, x = 't', y = 'Target_deg', hue = 'PROTAC', palette = "Set2")
# plt.xlim(t.min(), t.max())
# plt.ylim(0, 120)
# plt.xlabel('Time (h)')
# plt.ylabel('% Baseline Target Protein')
# plt.title('Percent baseline Target protein at [$BPD_{ec}$] = ' + f'{str(initial_BPD_ec_conc)} uM')
# plt.savefig(f'plots/Target_Deg_BPD={str(initial_BPD_ec_conc[0])}uM.png')
