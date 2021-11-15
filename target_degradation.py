import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import kinetic_proofreading as kp
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,8)

"""
PLOT OPTIMAL BPD CONCENTRATION OVER TIME
"""
Prop_Target_Deg_Grid = np.load('saved_objects/Prop_Target_Deg_Grid.npy')

max_degradation_idx = np.argmin(Prop_Target_Deg_Grid, axis = 0)
Conc_BPD_ec_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)  # various initial concentrations of BPD_ec
t = np.arange(start = 0, stop = 48 + 1, step = 2)  # time points at which to evaluate solver
max_deg_Conc_BPD_ec_arr = Conc_BPD_ec_arr[max_degradation_idx]

Target_deg_df = pd.DataFrame({'t': t, 'Target_deg': max_deg_Conc_BPD_ec_arr})

ax = Target_deg_df.plot(
    x = 't',
    xlabel = 'Time (h)',
    y = 'Target_deg',
    ylabel = '$BPD_{ec}$ Concentration (nM)',
    kind = 'line',
    xlim = (0, 48),
    xticks = np.arange(start = 0, stop = 48 + 1, step = 6),
    fontsize = 20,
    title='Optimal $BPD_{ec}$ Concentration over Time',
    legend = False,
    figsize = (12, 8)
)
plt.savefig('plots/Optimal_BPD_vs_Time')
plt.show()
