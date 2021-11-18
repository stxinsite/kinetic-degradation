import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import mplot3d
from tqdm import tqdm
import kinetic_module.kinetic_functions as kinetic_functions
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,8)

def initial_values(params, BPD_ec = 0, BPD_ic = 0):
    T = params['Conc_T_base'] * params['Vic']
    E3 = params['Conc_E3_base'] * params['Vic']
    BPD_T = 0
    BPD_E3 = 0
    Ternary = 0
    Ternary_Ubs = [0] * params['n']
    return np.array([BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)

def ternary_formation_fixed_time(params, t, initial_BPD_ic_conc, PROTAC_ID, save_plot = True):
    y0 = initial_values(params)
    y0[1] = conc * params['Vic'] / 1000  # set initial BPD_ic

def calc_degradation_curve(initial_BPD_ec_conc, t, params):
    """
    Calculates target protein degradation and ternary formation curves
    for fixed initial extracellular degrader concentration at time points t.

    Args:
        initial_BPD_ec_conc: float; initial value of BPD_ec concentration.
        t: array_like; time points at which to calculate.
        params: dict; passed to kinetic_functions.calc_concentrations().

    Returns:
        result: pd.DataFrame; percent degradation and ternary formation relative to baseline Target at time points
    """
    initial_BPD_ec = initial_BPD_ec_conc * params['Vec']
    y0 = initial_values(params, BPD_ec = initial_BPD_ec)
    concentrations = kinetic_functions.calc_concentrations(params, t, y0, max_step = 0.001)
    concentrations_df = kinetic_functions.dataframe_concentrations(concentrations)

    T_totals = concentrations_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)  # pd.Series: total amounts of Target at time points t
    Ternary_totals = concentrations_df['Ternary']  # pd.Series: amounts of un-ubiquitinated Ternary at time points t
    all_Ternary_totals = concentrations_df.filter(regex = 'Ternary.*').sum(axis = 1)  # pd.Series: total amounts of Ternary at time points t

    relative_T = T_totals / y0[2] * 100  # percent total Target relative to baseline Target
    relative_Ternary = Ternary_totals / y0[2] * 100  # percent Ternary relative to baseline Target
    relative_all_Ternary = all_Ternary_totals / y0[2] * 100  # percent total Ternary relative to baseline Target

    result = pd.DataFrame({
        't': t,
        'degradation': relative_T,
        'Ternary': relative_Ternary,
        'all_Ternary': relative_all_Ternary
    })
    return result

def degradation_vary_BPD_time(params, t, initial_BPD_ec_conc, PROTAC_ID, save_plot = True):
    y0 = initial_values(params)
    Prop_Target_Deg_Grid = np.empty( (len(initial_BPD_ec_conc), len(t)) )  # array to store percent relative Target protein degradation

    for count, conc in enumerate(tqdm(initial_BPD_ec_conc)):
        y0[0] = conc * params['Vec']  # set initial BPD_ec
        results = kinetic_functions.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)  # solve initial value problem
        results_df = kinetic_functions.dataframe_concentrations(results)
        T_totals = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)  # sum amounts of all complexes containing Target at each time point
        Target_deg = T_totals / y0[2] * 100  # divide total Target amount at each time point by initial value of Target
        assert np.all(Target_deg <= 101.), f"Relative degradation is greater than 100% at some time points."

        Prop_Target_Deg_Grid[count] = Target_deg

    np.save(f'saved_objects/Prop_Target_Deg_Grid_{PROTAC_ID}.npy', Prop_Target_Deg_Grid)  # save the solved relative Target degradation proportions

    if save_plot:
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

    return Prop_Target_Deg_Grid

def solve_target_degradation(initial_BPD_ec_concs, t, params, PROTAC_ID):
    """
    Calculates target protein degradation and ternary formation curves
    for various initial BPD_ec concentrations at time points t.

    Args:
        initial_BPD_ec_concs: array_like; initial values of BPD_ec concentrations.
        t: array_like; time points at which to calculate.
        params: dict; passed to kinetic_functions.calc_concentrations().
        PROTAC_ID: string; PROTAC identifier.
    """
    pool = mp.Pool(processes=mp.cpu_count())
    inputs = [(conc, t, params) for conc in initial_BPD_ec_concs]
    outputs = pool.starmap(calc_degradation_curve, inputs)
    pool.close()
    pool.join()
    result = pd.concat(outputs, ignore_index = True)
    return result
