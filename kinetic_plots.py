import numpy as np
import pandas as pd
from kinetic_proofreading import calc_concentrations, dataframe_concentrations, plot_concentrations
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import mplot3d
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,9)

"""SIMULATIONS"""
# species amounts at time = 0
y0 = np.array([100 * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
# time steps
t = np.arange(start = 0, stop = 48 + 1, step = 6)
# t = np.array([0, 24, 48])

results = calc_concentrations(times = t, y0 = y0, max_step = 0.001)
results.message
results.success
np.all(results.y >= 0)

results_df = dataframe_concentrations(results)

plot_concentrations(results_df)

BPD_total = results_df.filter(regex = '(BPD.*)|(Ternary.*)', axis = 1).sum(axis = 1)
np.allclose(BPD_total, BPD_total[0])

T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)
np.allclose(T_total, T_total[0])

E3_total = results_df.filter(regex = '(.*E3)|(Ternary.*)').sum(axis = 1)
np.allclose(E3_total, E3_total[0])

"""
RELATIVE PROTEIN DEGRADATION OVER TIME AND BPD CONCENTRATION
"""
Conc_BPD_ec_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)  # array of [BPD_ec]
t_arr = np.arange(start = 0, stop = 48 + 1, step = 2)  # array of time points
Prop_Target_Deg_Grid = np.empty((len(Conc_BPD_ec_arr), len(t_arr)))
Prop_Target_Deg_Grid.shape

%%time
for count, conc in enumerate(Conc_BPD_ec_arr):
    if (count + 1) % 10 == 0:
        progress = round((count + 1) / len(Conc_BPD_ec_arr) * 100)
        print(f"Progress: {progress}%")

    y0_arr = np.array([conc * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)  # initial values at conc: [BPD_ec]

    results = calc_concentrations(times = t_arr, y0 = y0_arr, max_step = 0.001)  # solve_ivp(max_step = max_step)
    assert np.all(results.y >= 0), f"Results from solve_ivp() at initial value [BPD_ec] = {conc} are not all non-negative."

    results_df = dataframe_concentrations(results)

    T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)  # sum amounts of all complexes containing Target at each time point
    Target_deg = T_total / T * 100  # divide total Target amount at each time point by initial value of Target
    assert np.all(Target_deg <= 100.), f"Relative degradation at initial value [BPD_ec] = {conc} is greater than 100%."

    Prop_Target_Deg_Grid[count] = Target_deg

# save the solved relative Target degradation proportions
np.save('Prop_Target_Deg_Grid.npy', Prop_Target_Deg_Grid)
Prop_Target_Deg_Grid = np.load('saved_objects/Prop_Target_Deg_Grid.npy')

# 3D grid of percent relative Target degradation
X, Y = np.meshgrid(t_arr, np.log10(Conc_BPD_ec_arr))
def Conc_BPD_ticks(x, pos):
    return '{:.0e}'.format(10 ** x)

fig = plt.figure(figsize = (9,9))
ax = fig.add_subplot(projection = '3d')
Prop_Target_Deg_Contour_F = ax.plot_surface(X, Y, Prop_Target_Deg_Grid, cmap = 'plasma')
ax.set_xlabel("Time (h)")
ax.set_ylabel("BPD Concentration (nM)")
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.zaxis.set_tick_params(labelsize=15)
ax.view_init(10, 30)
formatter = FuncFormatter(Conc_BPD_ticks)
ax.yaxis.set_major_formatter(formatter)
plt.savefig("Target_Deg_over_BPD_vs_Time_3D.png")
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
plt.savefig('Target_Deg_over_BPD_vs_Time')
plt.show()

"""
PLOT OPTIMAL BPD CONCENTRATION OVER TIME
"""
max_degradation_idx = np.argmin(Prop_Target_Deg_Grid, axis = 0)
max_deg_Conc_BPD_ec_arr = Conc_BPD_ec_arr[max_degradation_idx]

max_deg_Conc_BPD_ec_arr

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
plt.savefig('Optimal_BPD_vs_Time')
plt.show()


"""
BARTLETT SUPPLEMENTARY FIGURE 1 (c)
"""
y0 = np.array([100 * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
t = np.arange(start = 0, stop = 48 + 1, step = 2)

results = calc_concentrations(times = t, y0 = y0, max_step = 0.001)
np.all(results.y >= 0)

results_df = dataframe_concentrations(results)

T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)
Target_deg = T_total / T_total[0] * 100

Target_deg_df = pd.DataFrame({'t': t, 'Target_deg': Target_deg})

ax = Target_deg_df.plot(
    x = 't',
    xlabel = 'Time (h)',
    y = 'Target_deg',
    ylabel = '% Baseline Target Protein',
    kind = 'line',
    xlim = (0, 48),
    ylim = (0, 120),
    xticks = np.arange(start = 0, stop = 48 + 1, step = 6),
    fontsize = 20,
    # title='Ternary complex formation',
    legend = False,
    figsize = (12, 8)
)
plt.show()

"""
BARTLETT SUPPLEMENTARY FIGURE 1 (b)
"""
Conc_BPD_ec_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)
Target_deg_arr = np.empty((len(Conc_BPD_ec_arr),2))

# Takes ~100s for 20 concentrations, 2 time points each
# %%timeit
for count, conc in enumerate(Conc_BPD_ec_arr):
    if (count + 1) % 10 == 0:
        progress = (count + 1) / len(Conc_BPD_ec_arr) * 100
        print("Progress: " + str(progress))

    y0 = np.array([conc * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
    t = np.array([0, 24])

    results = calc_concentrations(times = t, y0 = y0, max_step = 0.0024)

    results_df = dataframe_concentrations(results)

    T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)

    Target_deg_arr[count, 0] = conc
    Target_deg_arr[count, 1] = T_total.values[1] / T * 100

Target_deg_df = pd.DataFrame(Target_deg_arr, columns = ['Conc_BPD_ec', 'Target_deg'])

ax = Target_deg_df.plot(
    x = 'Conc_BPD_ec',
    xlabel = 'BPD Concentration (nM)',
    y = 'Target_deg',
    ylabel = '% Baseline Target Protein',
    kind = 'line',
    xlim = (1e-1, 1e5),
    ylim = (0, 120),
    fontsize = 20,
    logx = True,
    # title='Ternary complex formation',
    legend = False,
    figsize = (12, 8)
)
plt.show()
