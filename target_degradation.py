import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import mplot3d
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
LOAD PARAMETERS FROM CONFIG
"""
with open('model_configs/degradation_config.yml', 'r') as file:
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
RELATIVE PROTEIN DEGRADATION OVER TIME AND BPD CONCENTRATION
"""
Conc_BPD_ec_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)  # various initial concentrations of BPD_ec
t_arr = np.arange(start = 0, stop = 48 + 1, step = 2)  # time points at which to evaluate solver
Prop_Target_Deg_Grid = np.empty((len(Conc_BPD_ec_arr), len(t_arr)))  # array to store percent relative Target protein degradation

for count, conc in enumerate(Conc_BPD_ec_arr):
    y0_arr = np.array([conc * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)  # initial values
    results = kp.calc_concentrations(params = params, times = t_arr, y0 = y0_arr, max_step = 0.001)  # solve initial value problem
    assert np.all(results.y >= 0), f"Results from solve_ivp() at initial value [BPD_ec] = {conc} are not all non-negative."

    results_df = kp.dataframe_concentrations(results)

    T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)  # sum amounts of all complexes containing Target at each time point
    Target_deg = T_total / T * 100  # divide total Target amount at each time point by initial value of Target
    assert np.all(Target_deg <= 100.), f"Relative degradation at initial value [BPD_ec] = {conc} is greater than 100%."

    Prop_Target_Deg_Grid[count] = Target_deg

    if (count + 1) % 10 == 0:
        progress = round((count + 1) / len(Conc_BPD_ec_arr) * 100)
        print(f"Progress: {progress}%")

np.save('Prop_Target_Deg_Grid.npy', Prop_Target_Deg_Grid)  # save the solved relative Target degradation proportions
# Prop_Target_Deg_Grid = np.load('saved_objects/Prop_Target_Deg_Grid.npy')  # or load previously saved results

# 3D grid of percent relative Target degradation
X, Y = np.meshgrid(t_arr, np.log10(Conc_BPD_ec_arr))
def Conc_BPD_ticks(x, pos):
    return '{:.0e}'.format(10 ** x)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
Prop_Target_Deg_Contour_F = ax.plot_surface(X, Y, Prop_Target_Deg_Grid, cmap = 'plasma')
ax.set_title("Percent baseline Target protein using BTK PROTAC parameters")
ax.set_xlabel("\nTime (h)")
ax.set_ylabel("\nBPD Concentration (nM)")
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.zaxis.set_tick_params(labelsize=15)
ax.view_init(25, -35)
formatter = FuncFormatter(Conc_BPD_ticks)
ax.yaxis.set_major_formatter(formatter)
plt.savefig("plots/Target_Deg_over_BPD_vs_Time_3D.png")
plt.show()

# 2D contour plot of percent relative Target degradation
contour_levels = np.linspace(start = 0, stop = 100, num = 20)

fig = plt.figure(figsize = (9,9))
Prop_Target_Deg_Contour_L = plt.contour(t_arr, Conc_BPD_ec_arr, Prop_Target_Deg_Grid, levels = contour_levels, colors = 'black', linewidths = 0.75)
Prop_Target_Deg_Contour_F = plt.contourf(t_arr, Conc_BPD_ec_arr, Prop_Target_Deg_Grid, levels = contour_levels, cmap = 'plasma')
norm = matplotlib.colors.Normalize(vmin = contour_levels.min(), vmax = contour_levels.max())
sm = plt.cm.ScalarMappable(norm = norm, cmap = Prop_Target_Deg_Contour_F.cmap)
sm.set_array([])
fig.colorbar(sm, ticks = np.linspace(start = 0, stop = 100, num = 11))
plt.xlim(np.min(t_arr), np.max(t_arr))
plt.ylim(np.min(Conc_BPD_ec_arr), np.max(Conc_BPD_ec_arr))
plt.title('% Baseline Target Protein')
plt.xlabel('Time (h)')
plt.ylabel('BPD Concentration (nM)')
plt.xticks(np.arange(t_arr.min(), t_arr.max() + 1, step = 6), fontsize = 15)
plt.yticks(fontsize = 15)
plt.yscale('log')
plt.savefig('plots/Target_Deg_over_BPD_vs_Time')
plt.show()

"""
PLOT OPTIMAL BPD CONCENTRATION OVER TIME
"""
max_degradation_idx = np.argmin(Prop_Target_Deg_Grid, axis = 0)
max_deg_Conc_BPD_ec_arr = Conc_BPD_ec_arr[max_degradation_idx]

Target_deg_df = pd.DataFrame({'t': t_arr, 'Target_deg': max_deg_Conc_BPD_ec_arr})

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
