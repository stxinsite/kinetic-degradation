import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import kinetic_proofreading as kp
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,8)

"""
This script reproduces Bartlett et al. (2013) Supplementary Figure 1a.

Given initial values, we solve for amounts of Ternary complex formed after 24 hours
at various concentrations of PROTAC.

Ternary complex formation with intracellular PROTAC only.
No ubiquitination or degradation.
"""

"""
LOAD PARAMETERS FROM CONFIG
"""
with open('model_configs/ternary_formation_config.yml', 'r') as file:
    params = yaml.safe_load(file)

"""
INITIAL VALUES
"""
BPD_ec = 0  # nM * Vec / 1000
BPD_ic = 0  # nM * Vic / 1000
T = params['Conc_T_base'] * params['Vic']
E3 = params['Conc_E3_base'] * params['Vic']
BPD_T = 0
BPD_E3 = 0
Ternary = 0
Ternary_Ubs = [0] * params['n']

"""
VARY BPD_ic CONCENTRATION
"""
Conc_BPD_ic_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)  # various initial concentrations of BPD_ic
Ternary_formation_arr = np.empty( (len(Conc_BPD_ic_arr), 2) )  # an array for storing Ternary amounts at concentrations of BPD_ic

for count, conc in enumerate(Conc_BPD_ic_arr):
    y0 = np.array([BPD_ec, conc * params['Vic'] / 1000, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)  # initial values
    t = np.array([0, 24])  # time points at which to evaluate solution to initial value problem
    results = kp.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)  # solve initial value problem
    assert np.all(results.y >= 0), f"Results from solve_ivp() at initial value [BPD_ic] = {conc} are not all non-negative."

    Ternary_formation_arr[count] = np.array([conc, results.y[6][-1]])  # BPD_ic concentration and latest Ternary amount

    if (count + 1) % 10 == 0:
        progress = round((count + 1) / len(Conc_BPD_ec_arr) * 100)
        print(f"Progress: {progress}%")

Ternary_formation_df = pd.DataFrame(Ternary_formation_arr, columns = ['Conc_BPD_ic', 'Ternary'])
# relative Ternary is percentage of the max Ternary amount across all concentrations of BPD_ic
Ternary_formation_df['relative_Ternary'] = Ternary_formation_df['Ternary'] / Ternary_formation_df['Ternary'].max() * 100

ax = Ternary_formation_df.plot(
    x = 'Conc_BPD_ic',
    xlabel = '$BPD_{ic}$ Concentration (nM)',
    y = 'relative_Ternary',
    ylabel = '% Relative Ternary Complex',
    kind = 'line',
    xlim = (Conc_BPD_ic_arr.min(), Conc_BPD_ic_arr.max()),
    ylim = (0, 120),
    fontsize = 20,
    logx = True,
    title='Ternary complex formation as function of BPD exposure\n using parameters from BTK PROTAC system',
    legend = False,
)
plt.savefig("plots/Relative_Ternary_Formation.png")
plt.show()
