"""
This module contains functions used to calculate target protein degradation, ternary complex formation, and Dmax for
various configurations of model parameters and initial values.
"""

from typing import Iterable, Optional
import yaml
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
# from ray.util.multiprocessing import Pool
import kinetic_module.kinetic_functions as kf
from kinetic_module.calc_full_config import KineticParameters
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


def solve_target_degradation(t_eval: ArrayLike,
                             params: dict[str, float],
                             initial_BPD_ec_concs: Optional[Iterable] = None,
                             initial_BPD_ic_concs: Optional[Iterable] = None,
                             return_only_final_state: bool = False,
                             PROTAC_ID: Optional[str] = None) -> pd.DataFrame:
    """Calculates target protein degradation, ternary complex formation, and Dmax
    for possible various initial concentrations of degrader over a range of time points.

    Parameters
    ----------
    t_eval : ArrayLike
        time points at which to store computed solution

    params : dict[str, float]
        kinetic rate constants and model parameters for rate equations

    initial_BPD_ec_concs : Optional[Iterable]
        initial value(s) of BPD_ec concentration

    initial_BPD_ic_concs : Optional[Iterable]
        initial value(s) of BPD_ic concentration

    return_only_final_state : bool
        whether to return only final state of system

    PROTAC_ID : Optional[str]
        PROTAC identifier

    Returns
    -------
    pd.DataFrame
        percent degradation, ternary formation relative to baseline total Target amount and
        percent Dmax for all initial concentrations of degrader over a range of time points
    """
    if initial_BPD_ec_concs is None and initial_BPD_ic_concs is None:
        print('No initial concentrations of degrader were provided.')
        return None
    elif initial_BPD_ec_concs is not None and initial_BPD_ic_concs is None:
        initial_BPD_ic_concs = np.zeros(len(initial_BPD_ec_concs))
    elif initial_BPD_ec_concs is None and initial_BPD_ic_concs is not None:
        initial_BPD_ec_concs = np.zeros(len(initial_BPD_ic_concs))
    else:
        assert len(initial_BPD_ec_concs) == len(initial_BPD_ic_concs)

    inputs = [
        (t_eval, params, initial_BPD_ec, initial_BPD_ic, return_only_final_state)
        for (initial_BPD_ec, initial_BPD_ic) in zip(initial_BPD_ec_concs, initial_BPD_ic_concs)
    ]

    if len(inputs) > 1:
        pool = Pool(processes=cpu_count())
        outputs = pool.starmap(kf.calc_degradation_curve, inputs)
        pool.close()
        pool.join()
    else:
        args = inputs[0]
        outputs = [kf.calc_degradation_curve(*args)]

    result = pd.concat(outputs, ignore_index=True)
    result['PROTAC'] = PROTAC_ID
    return result


def run_kinetic_model(config_files: list[str],
                      protac_IDs: list[str],
                      t_eval: ArrayLike = np.linspace(0, 1),
                      initial_BPD_ic_concs: Optional[Iterable] = None,
                      initial_BPD_ec_concs: Optional[Iterable] = None,
                      return_only_final_state: bool = False) -> pd.DataFrame:
    outputs = []
    for config, protac in zip(config_files, protac_IDs):
        params = get_params_from_config(config)

        df = solve_target_degradation(
            t_eval=t_eval,
            params=params,
            initial_BPD_ec_concs=initial_BPD_ec_concs,
            initial_BPD_ic_concs=initial_BPD_ic_concs,
            return_only_final_state=return_only_final_state,
            PROTAC_ID=protac
        )

        outputs.append(df)

    result = pd.concat(outputs, ignore_index=True)
    return result


def get_params_from_config(config_filename: str) -> dict[str, float]:
    # this will probably break if cwd is not kinetic-degradation
    with open(file=f'./data/{config_filename}', mode='r') as file:
        config_dict = yaml.safe_load(file)

    params = KineticParameters(config_dict)
    if params.is_fully_defined():
        result = params.params
    else:
        raise ValueError(f'Parameters in {config_filename} are insufficient or inconsistent.')

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
