import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.ticker import FuncFormatter
from mpl_toolkits import mplot3d
from tqdm import tqdm
import kinetic_module.kinetic_functions as kf
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12,8)

"""
To do:
- allow calc_degradation_curve() to set initial BPD_ic
    - this would allow for entirely intracellular systems
- parallelize solve_target_degradation().
    - since each element in inputs is independent, calc_degradation_curve()
      can be called in parallel
- solve_target_degradation() does same function as degradation_vary_BPD_time(),
  but shape of output is different
    - allow conversion of long pd.DataFrame to grid
        - x: BPD concentration
        - y: time points
        - z: percent relative Target degradation
    - to plot surface plot
"""

def solve_target_degradation(t_eval, params, initial_BPD_ec_concs=[], initial_BPD_ic_concs=[], return_only_final_state=True, PROTAC_ID=None):
    """
    Calculate target protein degradation and ternary formation
    for various initial degrader concentrations at time points t_eval.

    Arguments:
        t_eval: array_like; time points at which to store computed solution.
        params: dict; kinetic rate constants and model parameters for rate equations.
        initial_BPD_ec_concs: array_like; initial value of BPD_ec concentration.
        initial_BPD_ic_concs: array_like; initial value of BPD_ic concentration.
        return_only_final_state: bool; whether to return only final state of system.
        PROTAC_ID: str; PROTAC identifier

    Returns:
        pd.DataFrame; percent degradation and ternary formation relative to baseline Target
            at time points t_eval for all initial degrader concentrations.
    """
    assert pd.api.types.is_list_like(initial_BPD_ec_concs) and pd.api.types.is_list_like(initial_BPD_ic_concs)
    if initial_BPD_ec_concs and not initial_BPD_ic_concs:
        # initial_BPD_ec_concs is provided but initial_BPD_ic_concs is empty
        initial_BPD_ic_concs = [0] * len(initial_BPD_ec_concs)
    elif not initial_BPD_ec_concs and initial_BPD_ic_concs:
        # initial_BPD_ic_concs is provided but initial_BPD_ec_concs is empty
        initial_BPD_ec_concs = [0] * len(initial_BPD_ic_concs)
    else:
        # both or neither are provided
        assert len(initial_BPD_ec_concs) == len(initial_BPD_ic_concs)

    inputs = [
        (t_eval, params, initial_BPD_ec, initial_BPD_ic, return_only_final_state)
        for (initial_BPD_ec, initial_BPD_ic) in zip(initial_BPD_ec_concs, initial_BPD_ic_concs)
    ]

    if len(initial_BPD_ec_concs) > 1:
        pool = mp.Pool(processes=mp.cpu_count())
        outputs = pool.starmap(kf.calc_degradation_curve, inputs)
        pool.close()
        pool.join()
    else:
        outputs = [kf.calc_degradation_curve(*args) for args in inputs]

    result = pd.concat(outputs, ignore_index=True)
    result['PROTAC'] = PROTAC_ID
    return result


"""
DO NOT USE YET. WILL NOT WORK.
"""
# def degradation_vary_BPD_time(params, t, initial_BPD_ec_conc, PROTAC_ID, save_plot = True):
#     y0 = initial_values(params)
#     T_total_baseline = np.sum(np.concatenate((y0[[2,4]], y0[6:])))
#     Prop_Target_Deg_Grid = np.empty( (len(initial_BPD_ec_conc), len(t)) )  # array to store percent relative Target protein degradation
#
#     for count, conc in enumerate(tqdm(initial_BPD_ec_conc)):
#         y0[0] = conc * params['Vec']  # set initial BPD_ec
#         results = kf.calc_concentrations(params = params, times = t, y0 = y0, max_step = 0.001)  # solve initial value problem
#         results_df = kf.dataframe_concentrations(results, params['n'])
#         T_totals = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)  # sum amounts of all complexes containing Target at each time point
#         Target_deg = T_totals / T_total_baseline * 100  # divide total Target amount at each time point by initial value of Target
#         assert np.all(Target_deg <= 101.), f"Relative degradation is greater than 100% at some time points."
#
#         Prop_Target_Deg_Grid[count] = Target_deg
#
#         np.save(f'saved_objects/Prop_Target_Deg_Grid_{PROTAC_ID}.npy', Prop_Target_Deg_Grid)  # save the solved relative Target degradation proportions
#
#         if save_plot:
#             contour_levels = np.linspace(start = 0, stop = 100, num = 20)  # 2D contour plot of percent relative Target degradation
#
#             fig = plt.figure(figsize = (9,9))
#             Prop_Target_Deg_Contour_L = plt.contour(t, initial_BPD_ec_conc, Prop_Target_Deg_Grid, levels = contour_levels, colors = 'black', linewidths = 0.75)
#             Prop_Target_Deg_Contour_F = plt.contourf(t, initial_BPD_ec_conc, Prop_Target_Deg_Grid, levels = contour_levels, cmap = 'plasma')
#             norm = matplotlib.colors.Normalize(vmin = contour_levels.min(), vmax = contour_levels.max())
#             sm = plt.cm.ScalarMappable(norm = norm, cmap = Prop_Target_Deg_Contour_F.cmap)
#             sm.set_array([])
#             fig.colorbar(sm, ticks = np.linspace(start = 0, stop = 100, num = 11))
#             plt.xlim(np.min(t), np.max(t))
#             plt.ylim(np.min(initial_BPD_ec_conc), np.max(initial_BPD_ec_conc))
#             plt.title(f'% Baseline Target Protein with {PROTAC_ID}')
#             plt.xlabel('Time (h)')
#             plt.ylabel('BPD Concentration (uM)')
#             plt.xticks(np.arange(t.min(), t.max() + 1, step = 6), fontsize = 15)
#             plt.yticks(fontsize = 15)
#             plt.yscale('log')
#             plt.savefig(f'plots/Target_Deg_{PROTAC_ID}_contour.png')
#
#             return Prop_Target_Deg_Grid
