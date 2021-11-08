import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import kinetic_proofreading as kp
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,8)

"""
This script reproduces Bartlett et al. (2013) Supplementary Figures 1b and 1c.

Given initial values, we solve for target protein degradation relative to initial baseline amount
at various concentrations of PROTAC.
"""

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


"""
BARTLETT SUPPLEMENTARY FIGURE 1 (c)
"""
y0 = np.array([100 * params['Vec'] / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
t = np.arange(start = 0, stop = 48 + 1, step = 2)

results = kp.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)
assert np.all(results.y >= 0), f"Results from solve_ivp() at initial value [BPD_ec] = {conc} are not all non-negative."

results_df = kp.dataframe_concentrations(results)

T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)
Target_deg = T_total / T_total[0] * 100

Target_deg_df = pd.DataFrame({'t': t, 'Target_deg': Target_deg})

ax = Target_deg_df.plot(
    x = 't',
    xlabel = 'Time (h)',
    y = 'Target_deg',
    ylabel = '% Baseline Target Protein',
    kind = 'line',
    xlim = (t.min(), t.max()),
    ylim = (0, 120),
    xticks = np.arange(start = t.min(), stop = t.max() + 1, step = 6),
    fontsize = 20,
    title='Percent baseline Target protein over time\n at [$BPD_{ec}$] = 100 nM using BTK PROTAC parameters',
    legend = False,
    figsize = (12, 8)
)
plt.savefig("plots/Target_Deg_BPD=100nM.png")
plt.show()

"""
BARTLETT SUPPLEMENTARY FIGURE 1 (b)
"""
Conc_BPD_ec_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)
Target_deg_arr = np.empty((len(Conc_BPD_ec_arr), 2))

# Takes ~100s for 20 concentrations, 2 time points each
for count, conc in enumerate(Conc_BPD_ec_arr):
    y0 = np.array([conc * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
    t = np.array([0, 24])

    results = kp.calc_concentrations(times = t, y0 = y0, max_step = 0.001)
    assert np.all(results.y >= 0), f"Results from solve_ivp() at initial value [BPD_ec] = {conc} are not all non-negative."

    results_df = kp.dataframe_concentrations(results)

    T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)

    Target_deg_arr[count] = np.array([conc, T_total.values[1] / T * 100])

    if (count + 1) % 10 == 0:
        progress = (count + 1) / len(Conc_BPD_ec_arr) * 100
        print("Progress: " + str(progress))

Target_deg_df = pd.DataFrame(Target_deg_arr, columns = ['Conc_BPD_ec', 'Target_deg'])

ax = Target_deg_df.plot(
    x = 'Conc_BPD_ec',
    xlabel = 'BPD Concentration (nM)',
    y = 'Target_deg',
    ylabel = '% Baseline Target Protein',
    kind = 'line',
    xlim = (Conc_BPD_ec_arr.min(), Conc_BPD_ec_arr.max()),
    ylim = (0, 120),
    fontsize = 20,
    logx = True,
    title='Percent baseline Target protein at 24 hours using BTK PROTAC parameters',
    legend = False,
    figsize = (12, 8)
)
plt.savefig('plots/Target_Deg_t=24H.png')
plt.show()
