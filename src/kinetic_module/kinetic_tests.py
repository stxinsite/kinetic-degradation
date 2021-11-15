import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import mplot3d
from tqdm import tqdm
import kinetic_module.kinetic_functions as kinetic_functions
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,8)

def initial_values(params):
    BPD_ec = 0  # nM * Vec / 1000
    BPD_ic = 0  # nM * Vic / 1000
    T = params['Conc_T_base'] * params['Vic']
    E3 = params['Conc_E3_base'] * params['Vic']
    BPD_T = 0
    BPD_E3 = 0
    Ternary = 0
    Ternary_Ubs = [0] * params['n']
    return np.array([BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)

def solve_ternary_formation(params):
    y0 = initial_values(params)
    t = np.array([0, 24])  # time points at which to evaluate solution to initial value problem

    Conc_BPD_ic_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)  # various initial concentrations of BPD_ic
    Ternary_formation_arr = np.empty( (len(Conc_BPD_ic_arr), 2) )  # an array for storing Ternary amounts at concentrations of BPD_ic

    for count, conc in enumerate(tqdm(Conc_BPD_ic_arr)):
        y0[1] = conc * params['Vic'] / 1000  # set initial BPD_ic
        results = kinetic_functions.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)  # solve initial value problem
        assert np.all(results.y >= 0), f"Results from solve_ivp() at initial value [BPD_ic] = {conc} are not all non-negative."

        Ternary_formation_arr[count] = np.array([conc, results.y[6][-1]])  # BPD_ic concentration and latest Ternary amount

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

def degradation_fixed_BPD(params, t, initial_BPD_ec_conc, PROTAC_ID):
    """
    Solve target protein degradation for fixed initial BPD_ec at time points t.

    Args:
        params: dict; key-value pairs of all required parameters.
        t: array_like; time points at which to evaluate the solver.
        initial_BPD_ec_conc: float; initial value of BPD_ec concentration.
        PROTAC_ID: string; PROTAC identifier.

    Returns:
        Target_deg_df: DataFrame; percent Target degradation relative to baseline at time points
    """
    y0 = initial_values(params)
    y0[0] = initial_BPD_ec_conc * params['Vec']  # set initial BPD_ec
    results = kinetic_functions.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)  # solve initial value problem
    results_df = kinetic_functions.dataframe_concentrations(results)

    T_totals = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)  # sum amounts of all complexes containing Target at each time point
    Target_deg = T_totals / y0[2] * 100  # divide total Target amount at each time point by initial value of Target
    assert np.all(Target_deg <= 100.), f"Relative degradation is greater than 100% at some time points."

    Target_deg_df = pd.DataFrame({'t': t, 'Target_deg': Target_deg})

    ax = Target_deg_df.plot(
        x = 't',
        xlabel = 'Time (h)',
        y = 'Target_deg',
        ylabel = '% Baseline Target Protein',
        kind = 'line',
        xlim = (t.min(), t.max()),
        ylim = (0, 120),
        xticks = np.arange(start = t.min(), stop = t.max() + 1, step = 24),
        fontsize = 20,
        title=f'Percent baseline Target protein over time\n with {PROTAC_ID}' + ' at [$BPD_{ec}$] = ' + f'{str(initial_BPD_ec_conc)} uM',
        legend = False,
        figsize = (12, 8)
    )
    plt.savefig(f"plots/Target_Deg_{PROTAC_ID}_BPD_ec={str(initial_BPD_ec_conc)}uM.png")
    plt.show()
    return Target_deg_df

def degradation_fixed_time(params, t, initial_BPD_ec_conc, PROTAC_ID):
    y0 = initial_values(params)
    Target_deg_arr = np.empty( (len(initial_BPD_ec_conc), 2) )

    for count, conc in enumerate(tqdm(initial_BPD_ec_conc)):
        y0[0] = conc * params['Vec']  # set initial BPD_ec
        results = kinetic_functions.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)
        results_df = kinetic_functions.dataframe_concentrations(results)
        T_totals = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)
        Target_deg = T_totals / y0[2] * 100  # divide total Target amount by initial value of Target
        # print(Target_deg)  ### DELETE LATER AFTER TESTING ###
        assert np.all(Target_deg <= 100.), f"Relative degradation is greater than 100% at some time points."
        Target_deg_arr[count] = np.array( [conc, Target_deg.values[0]] )

    Target_deg_df = pd.DataFrame(Target_deg_arr, columns = ['Conc_BPD_ec', 'Target_deg'])

    ax = Target_deg_df.plot(
        x = 'Conc_BPD_ec',
        xlabel = 'BPD Concentration (uM)',
        y = 'Target_deg',
        ylabel = '% Baseline Target Protein',
        kind = 'line',
        xlim = (initial_BPD_ec_conc.min(), initial_BPD_ec_conc.max()),
        ylim = (0, 120),
        fontsize = 20,
        logx = True,
        title=f'Percent baseline Target protein at {str(t[0])} hours\n with {PROTAC_ID}',
        legend = False,
        figsize = (12, 8)
    )
    plt.savefig(f'plots/Target_Deg_{PROTAC_ID}_t={str(t[0])}h.png')
    plt.show()
    return Target_deg_df

def degradation_vary_BPD_time(params, t, initial_BPD_ec_conc, PROTAC_ID):
    y0 = initial_values(params)
    Prop_Target_Deg_Grid = np.empty( (len(initial_BPD_ec_conc), len(t)) )  # array to store percent relative Target protein degradation

    for count, conc in enumerate(tqdm(initial_BPD_ec_conc)):
        y0[0] = conc * params['Vec']  # set initial BPD_ec
        results = kinetic_functions.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)  # solve initial value problem
        results_df = kinetic_functions.dataframe_concentrations(results)
        T_totals = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)  # sum amounts of all complexes containing Target at each time point
        Target_deg = T_totals / y0[2] * 100  # divide total Target amount at each time point by initial value of Target
        assert np.all(Target_deg <= 100.), f"Relative degradation is greater than 100% at some time points."

        Prop_Target_Deg_Grid[count] = Target_deg

    np.save(f'saved_objects/Prop_Target_Deg_Grid_{PROTAC_ID}.npy', Prop_Target_Deg_Grid)  # save the solved relative Target degradation proportions

    contour_levels = np.linspace(start = 0, stop = 100, num = 20)  # 2D contour plot of percent relative Target degradation

    fig = plt.figure(figsize = (9,9))
    Prop_Target_Deg_Contour_L = plt.contour(t, initial_BPD_ec_conc, Prop_Target_Deg_Grid, levels = contour_levels, colors = 'black', linewidths = 0.75)
    Prop_Target_Deg_Contour_F = plt.contourf(t, initial_BPD_ec_conc, Prop_Target_Deg_Grid, levels = contour_levels, cmap = 'plasma')
    norm = matplotlib.colors.Normalize(vmin = contour_levels.min(), vmax = contour_levels.max())
    sm = plt.cm.ScalarMappable(norm = norm, cmap = Prop_Target_Deg_Contour_F.cmap)
    sm.set_array([])
    fig.colorbar(sm, ticks = np.linspace(start = 0, stop = 100, num = 11))
    plt.xlim(np.min(t), np.max(t))
    plt.ylim(np.min(initial_BPD_ec_conc), np.max(initial_BPD_ec_conc))
    plt.title(f'% Baseline Target Protein with {PROTAC_ID}')
    plt.xlabel('Time (h)')
    plt.ylabel('BPD Concentration (uM)')
    plt.xticks(np.arange(t.min(), t.max() + 1, step = 6), fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.yscale('log')
    plt.savefig(f'plots/Target_Deg_{PROTAC_ID}_contour.png')
    plt.show()
    return Prop_Target_Deg_Grid

def solve_target_degradation(params, t, initial_BPD_ec_conc, PROTAC_ID):
    """
    RELATIVE PROTEIN DEGRADATION OVER TIME AND BPD CONCENTRATION

    Args:
        params: dict; key-value pairs of all required parameters.
        t: array_like; time points at which to evaluate the solver.
        initial_BPD_ec: array_like; initial values of BPD_ec concentration.
        PROTAC_ID: string; PROTAC identifier.
    """
    if len(initial_BPD_ec_conc) == 1:
        return degradation_fixed_BPD(params, t, initial_BPD_ec_conc[0], PROTAC_ID)
    elif len(t) == 1:
        return degradation_fixed_time(params, t, initial_BPD_ec_conc, PROTAC_ID)
    else:
        return degradation_vary_BPD_time(params, t, initial_BPD_ec_conc, PROTAC_ID)
