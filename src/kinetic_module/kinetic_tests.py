"""
This module contains functions used to run a kinetic proofreading model of target protein degradation
using configuration(s) of model parameters and initial values.


"""
from typing import Iterable, Union, Optional
from multiprocessing import Pool, cpu_count

import yaml
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import matplotlib.pyplot as plt

# from ray.util.multiprocessing import Pool
import kinetic_module.kinetic_functions as kf
from kinetic_module.calc_full_config import KineticParameters

plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["figure.figsize"] = (12, 8)

"""
To do:
- solve_target_degradation() does same function as degradation_vary_BPD_time(),
  but shape of output is different
    - allow conversion of long pd.DataFrame to grid
        - x: BPD concentration
        - y: time points
        - z: percent relative Target degradation
    - to plot surface plot
"""


def solve_target_degradation(t_eval: Union[ArrayLike, float, int],
                             params: dict[str, float],
                             initial_BPD_ec_concs: Optional[Union[ArrayLike, float, int]] = None,
                             initial_BPD_ic_concs: Optional[Union[ArrayLike, float, int]] = None,
                             return_only_final_state: bool = False,
                             PROTAC_ID: Optional[str] = None) -> pd.DataFrame:
    """Calculates target protein degradation, ternary complex formation, and Dmax over time
    for initial concentration(s) of degrader.

    Parameters
    ----------
    t_eval : Union[ArrayLike, float, int]
        Time points at which to store computed solution.

    params : dict[str, float]
        Kinetic rate constants and model parameters for rate equations.

    initial_BPD_ec_concs : Optional[Union[ArrayLike, float, int]]
        Initial value(s) of BPD_ec concentration.

    initial_BPD_ic_concs : Optional[Union[ArrayLike, float, int]]
        Initial value(s) of BPD_ic concentration.

    return_only_final_state : bool
        Whether to return only final state of system.

    PROTAC_ID : Optional[str]
        PROTAC identifier.

    Returns
    -------
    pd.DataFrame
        Solutions at time points and initial configurations.

        ===================  ============================================================================
        t                    time point
        initial_BPD_ec_conc  initial extracellular BPD concentration
        initial_BPD_ic_conc  initial intracellular BPD concentration
        degradation          percent target protein degradation relative to baseline total Target
        Ternary              percent naked ternary complex formation relative to baseline total Target
        all_Ternary          percent all ternary complex formation relative to baseline total Target
        Dmax                 percent maximal target protein degradation relative to baseline total Target
        PROTAC_ID            PROTAC identifier
        Kd_T_binary          equilibrium dissociation constant of BPD-T binary complex
        kon_T_binary         kon of BPD + T -> BPD-T
        kub                  ternary complex ubiquitination rate
        alpha                cooperativity
        ===================  ============================================================================

    """

    if isinstance(t_eval, float) or isinstance(t_eval, int):
        t_eval = np.linspace(start=0, stop=t_eval)
    if isinstance(initial_BPD_ec_concs, float) or isinstance(initial_BPD_ec_concs, int):
        initial_BPD_ec_concs = np.array([initial_BPD_ec_concs])
    if isinstance(initial_BPD_ic_concs, float) or isinstance(initial_BPD_ic_concs, int):
        initial_BPD_ic_concs = np.array([initial_BPD_ic_concs])

    if initial_BPD_ec_concs is None and initial_BPD_ic_concs is None:
        print('No initial concentration(s) of degrader were provided.')
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
    outputs: list[pd.DataFrame]

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
    result['Kd_T_binary'] = params['Kd_T_binary']
    result['kon_T_binary'] = params['kon_T_binary']
    result['kub'] = params['kub']
    result['kde_ub'] = params['kde_ub']
    result['kdeg_UPS'] = params['kdeg_UPS']
    result['kdeg_Ternary'] = params['kdeg_Ternary']
    result['alpha'] = params['alpha']
    return result


def run_kinetic_model(config_files: list[str],
                      protac_IDs: list[str],
                      t_eval: Union[ArrayLike, float, int] = np.linspace(0, 1),
                      initial_BPD_ec_concs: Optional[Union[ArrayLike, float, int]] = None,
                      initial_BPD_ic_concs: Optional[Union[ArrayLike, float, int]] = None,
                      return_only_final_state: bool = False) -> pd.DataFrame:
    """Runs kinetic model for each pair of initial concentrations of extracellular and
    intracellular degrader for each configuration and PROTAC provided.

    Parameters
    ----------
    config_files : list[str]
        config filenames

    protac_IDs : list[str]
        PROTAC identifiers

    t_eval : Union[ArrayLike, float, int]
        time points at which to store state of system

    initial_BPD_ec_concs : Optional[Union[ArrayLike, float, int]]
        initial concentration(s) of extracellular degrader

    initial_BPD_ic_concs : Optional[Union[ArrayLike, float, int]]
        initial concentration(s) of intracellular degrader

    return_only_final_state : bool
        whether to return only final state of system

    Returns
    -------
    pd.DataFrame
        result returned by solve_target_degradation() for each configuration and initial concentration(s).
    """
    outputs: list[pd.DataFrame] = []

    for config, protac in zip(config_files, protac_IDs):
        params: dict[str, float] = get_params_from_config(config)

        df: pd.DataFrame = solve_target_degradation(
            t_eval=t_eval,
            params=params,
            initial_BPD_ec_concs=initial_BPD_ec_concs,
            initial_BPD_ic_concs=initial_BPD_ic_concs,
            return_only_final_state=return_only_final_state,
            PROTAC_ID=protac
        )
        outputs.append(df)

    result: pd.DataFrame = pd.concat(outputs, ignore_index=True)
    return result


def kd_T_binary_vs_alpha(config_filename: str,
                         protac_id: str,
                         t_eval: Union[ArrayLike, float, int],
                         alpha_range: Iterable[float],
                         kd_T_binary_range: Iterable[float],
                         initial_BPD_ec_conc: float = None,
                         initial_BPD_ic_conc: float = None,
                         ) -> pd.DataFrame:
    """Runs kinetic proofreading model for combinations of alpha and Kd_T_binary.

    Parameters
    ----------
    config_filename : str
        Config filename.

    protac_id : str
        PROTAC identifier.

    t_eval : Union[ArrayLike, float, int]
        Time points at which to store compute solution.

    alpha_range : Iterable[float]
        Range of cooperativity values.

    kd_T_binary_range : Iterable[float]
        Range of Kd_T_binary values.

    initial_BPD_ec_conc
        Initial concentration of extracellular BPD.

    initial_BPD_ic_conc
        Initial concentration of intracellular BPD.

    Returns
    -------
    pd.DataFrame
        Result returned by solve_target_degradation() for each (Kd_T_binary, alpha) and initial concentrations.
    """

    # these parameters will be set to None in order to be calculated and updated by KineticParameters()
    keys_to_update = [
        'koff_T_binary',
        'koff_T_ternary',
        'koff_E3_ternary',
        'Kd_T_ternary',
        'Kd_E3_ternary'
    ]
    result = run_alpha_across_param_levels(
        config_filename=config_filename,
        protac_id=protac_id,
        parameters_to_calc=keys_to_update,
        other_param_name='Kd_T_binary',
        other_param_range=kd_T_binary_range,
        alpha_range=alpha_range,
        t_eval=t_eval,
        initial_bpd_ec_conc=initial_BPD_ec_conc,
        initial_bpd_ic_conc=initial_BPD_ic_conc
    )
    return result


def kub_vs_alpha(config_filename: str,
                 protac_id: str,
                 t_eval: Union[ArrayLike, float, int],
                 alpha_range: Iterable[float],
                 kub_range: Iterable[float],
                 initial_BPD_ec_conc: float = None,
                 initial_BPD_ic_conc: float = None) -> pd.DataFrame:
    """Runs kinetic proofreading model for combinations of alpha and kub.

    Parameters
    ----------
    config_filename : str
        Config filename.

    protac_id : str
        PROTAC identifier.

    t_eval : Union[ArrayLike, float, int]
        Time points at which to store compute solution.

    alpha_range : ArrayLike
        Range of cooperativity values.

    kub_range
        Range of kub values.

    initial_BPD_ec_conc
        Initial concentration of extracellular BPD.

    initial_BPD_ic_conc
        Initial concentration of intracellular BPD.

    Returns
    -------
    pd.DataFrame
        result returned by solve_target_degradation() for each (kub, alpha) and initial concentrations.
    """

    # these parameters will be set to None in order to be calculated and updated by KineticParameters()
    keys_to_update = [
        'koff_T_binary',
        'koff_T_ternary',
        'koff_E3_binary',
        'koff_E3_ternary',
        'Kd_T_ternary',
        'Kd_E3_ternary'
    ]
    result = run_alpha_across_param_levels(
        config_filename=config_filename,
        protac_id=protac_id,
        parameters_to_calc=keys_to_update,
        other_param_name='kub',
        other_param_range=kub_range,
        alpha_range=alpha_range,
        t_eval=t_eval,
        initial_bpd_ec_conc=initial_BPD_ec_conc,
        initial_bpd_ic_conc=initial_BPD_ic_conc
    )
    return result


def kde_ub_vs_alpha(config_filename: str,
                    protac_id: str,
                    t_eval: Union[ArrayLike, float, int],
                    alpha_range: Iterable[float],
                    kde_ub_range: Iterable[float],
                    initial_BPD_ec_conc: float = None,
                    initial_BPD_ic_conc: float = None) -> pd.DataFrame:
    # these parameters will be set to None in order to be calculated and updated by KineticParameters()
    keys_to_update = [
        'koff_T_binary',
        'koff_T_ternary',
        'koff_E3_binary',
        'koff_E3_ternary',
        'Kd_T_ternary',
        'Kd_E3_ternary'
    ]
    result = run_alpha_across_param_levels(
        config_filename=config_filename,
        protac_id=protac_id,
        parameters_to_calc=keys_to_update,
        other_param_name='kde_ub',
        other_param_range=kde_ub_range,
        alpha_range=alpha_range,
        t_eval=t_eval,
        initial_bpd_ec_conc=initial_BPD_ec_conc,
        initial_bpd_ic_conc=initial_BPD_ic_conc
    )
    return result


def kdeg_ups_vs_alpha(config_filename: str,
                      protac_id: str,
                      t_eval: Union[ArrayLike, float, int],
                      alpha_range: Iterable[float],
                      kdeg_UPS_range: Iterable[float],
                      initial_BPD_ec_conc: float = None,
                      initial_BPD_ic_conc: float = None) -> pd.DataFrame:
    # these parameters will be set to None in order to be calculated and updated by KineticParameters()
    keys_to_update = [
        'koff_T_binary',
        'koff_T_ternary',
        'koff_E3_binary',
        'koff_E3_ternary',
        'Kd_T_ternary',
        'Kd_E3_ternary'
    ]
    result = run_alpha_across_param_levels(
        config_filename=config_filename,
        protac_id=protac_id,
        parameters_to_calc=keys_to_update,
        other_param_name='kdeg_UPS',
        other_param_range=kdeg_UPS_range,
        alpha_range=alpha_range,
        t_eval=t_eval,
        initial_bpd_ec_conc=initial_BPD_ec_conc,
        initial_bpd_ic_conc=initial_BPD_ic_conc
    )
    return result


def kdeg_ternary_vs_alpha(config_filename: str,
                          protac_id: str,
                          t_eval: Union[ArrayLike, float, int],
                          alpha_range: Iterable[float],
                          kdeg_Ternary_range: Iterable[float],
                          initial_BPD_ec_conc: float = None,
                          initial_BPD_ic_conc: float = None) -> pd.DataFrame:
    # these parameters will be set to None in order to be calculated and updated by KineticParameters()
    keys_to_update = [
        'koff_T_binary',
        'koff_T_ternary',
        'koff_E3_binary',
        'koff_E3_ternary',
        'Kd_T_ternary',
        'Kd_E3_ternary'
    ]
    result = run_alpha_across_param_levels(
        config_filename=config_filename,
        protac_id=protac_id,
        parameters_to_calc=keys_to_update,
        other_param_name='kdeg_Ternary',
        other_param_range=kdeg_Ternary_range,
        alpha_range=alpha_range,
        t_eval=t_eval,
        initial_bpd_ec_conc=initial_BPD_ec_conc,
        initial_bpd_ic_conc=initial_BPD_ic_conc
    )
    return result


def run_alpha_across_param_levels(config_filename: str,
                                  protac_id: str,
                                  parameters_to_calc: list[str],
                                  other_param_name: str,
                                  other_param_range: Iterable[float],
                                  alpha_range: Iterable[float],
                                  t_eval: Union[ArrayLike, float, int],
                                  initial_bpd_ec_conc: float,
                                  initial_bpd_ic_conc: float
                                  ) -> pd.DataFrame:
    """Runs kinetic proofreading model for combinations of alpha and another parameter.

    Parameters
    ----------
    config_filename : str
        Config filename.

    protac_id : str
        PROTAC identifier.

    parameters_to_calc : list[str]
        Names of parameters to re-calculate using new alpha and other parameter values.

    other_param_name : str
        Name of other parameter with which to pair with alpha.

    other_param_range : Iterable[float]
        Range of other parameter values with which to pair with alpha.

    alpha_range : Iterable[float]
        Range of alpha values with which to pair with other parameter.

    t_eval : Union[ArrayLike, float, int]
        Kinetic model solution at last time point will be stored.

    initial_bpd_ec_conc : float
        Initial concentration of extracellular BPD.

    initial_bpd_ic_conc : float
        Initial concentration of intracellular BPD.

    Returns
    -------
    pd.DataFrame
        Kinetic model solution for each alpha across another parameter levels.
    """
    params = get_params_from_config(config_filename=config_filename)
    params = set_keys_to_none(params, keys=parameters_to_calc)

    # combinations of alpha across levels of other parameter
    params_range = [
        (other_param_level, alpha)
        for other_param_level in other_param_range
        for alpha in alpha_range
    ]

    # list of parameters dictionaries updated for each alpha across each level of other parameter
    new_params: list[dict[str, float]] = copy_params(
        params=params,
        parameter_names=[other_param_name, 'alpha'],
        new_values=params_range
    )

    result = pool_solve_target_degradation(
        t_eval=t_eval,
        params_list=new_params,
        initial_bpd_ec_conc=initial_bpd_ec_conc,
        initial_bpd_ic_conc=initial_bpd_ic_conc,
        return_only_final_state=True,
        protac_id=protac_id
    )
    return result


def pool_solve_target_degradation(t_eval: Union[ArrayLike, float, int],
                                  params_list: list[dict[str, float]],
                                  initial_bpd_ec_conc: float,
                                  initial_bpd_ic_conc: float,
                                  return_only_final_state: bool,
                                  protac_id: str) -> pd.DataFrame:
    """Calls solve_target_degradation() in parallel for multiple parameter dictionaries.

    Parameters
    ----------
    t_eval : Union[ArrayLike, float, int]
        Time points at which to store computed solution.

    params_list : list[dict[str, float]]
        Multiple sets of kinetic rate constants and model parameters for rate equations.

    initial_bpd_ec_conc : Optional[Union[ArrayLike, float, int]]
        Initial value of BPD_ec concentration.

    initial_bpd_ic_conc : Optional[Union[ArrayLike, float, int]]
        Initial value of BPD_ic concentration.

    return_only_final_state : bool
        Whether to return only final state of system.

    protac_id : str
        PROTAC identifier.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of results from each parameter dictionary.
    """
    inputs = [
        (t_eval, params, initial_bpd_ec_conc, initial_bpd_ic_conc, return_only_final_state, protac_id)
        for params in params_list
    ]

    pool = Pool(processes=cpu_count())
    outputs = pool.starmap(solve_target_degradation, inputs)
    pool.close()
    pool.join()

    result = pd.concat(outputs)
    return result


def copy_params(params: dict[str, float],
                parameter_names: Iterable[str],
                new_values: list[tuple[float, ...]]) -> list[dict[str, float]]:
    """Updates parameters for each set of new values.

    Parameters
    ----------
    params : dict[str, float]
        Kinetic rate constants and model parameters.

    new_values : list[tuple[float]]
        List of new parameter values.

    parameter_names : list[str]
        Names of parameters to set to new values.

    Returns
    -------
    list[dict[str, float]]
        Updated parameter dictionaries re-calculated with new values.
    """
    params_copies: list[dict[str, float]] = [params.copy() for _ in new_values]
    result = [
        update_params(params=params_copy, keys=parameter_names, values=values)
        for params_copy, values in zip(params_copies, new_values)
    ]
    return result


def get_params_from_dict(params_dict: dict[str, float]) -> Optional[dict[str, float]]:
    """Returns valid dictionary of parameters if possible.

    Parameters
    ----------
    params_dict : dict[str, float]
        Kinetic rate constants and model parameters for rate equations.

    Returns
    -------
    Optional[dict[str, float]]
        A fully defined parameters dictionary if `params_dict` is sufficient and consistent. Otherwise, None.
    """
    params = KineticParameters(params_dict)
    if params.is_fully_defined():
        return params.params
    else:
        return None


def get_params_from_config(config_filename: str) -> Optional[dict[str, float]]:
    """Reads a config of model parameters and returns a dictionary.

    Parameters
    ----------
    config_filename : str
        Path to config file.

    Returns
    -------
    Optional[dict[str, float]]
        A fully defined dictionary of kinetic rate constants and model parameters for rate equations.

    Raises
    ------
    ValueError
        If config is insufficient or inconsistent.
    """
    split_config_filename = config_filename.split(sep='.')
    assert split_config_filename[-1] in ['yml', 'yaml'], 'File extension must be `.yml` or `.yaml`.'

    # this will probably break if cwd is not kinetic-degradation
    with open(file=f'./data/{config_filename}', mode='r') as file:
        config_dict: dict = yaml.safe_load(file)

    params = get_params_from_dict(config_dict)
    if params is None:
        raise ValueError(f'Parameters in {config_filename} are insufficient or inconsistent.')
    else:
        return params


def set_keys_to_none(a_dict: dict, keys: Iterable) -> dict:
    """Sets keys in dictionary to None.

    Parameters
    ----------
    a_dict : dict
        A dictionary containing all keys in `keys`.

    keys : Iterable
        Keys in dictionary whose corresponding values to set to None.

    Returns
    -------
    dict
        Dictionary with values corresponding to all keys in `keys` set to None.
    """
    for key in keys:
        a_dict[key] = None

    return a_dict


def update_params(params: dict[str, float], keys: Iterable[str], values: Iterable[float]) -> Optional[dict[str, float]]:
    """Updates a dictionary of kinetic rate constants and model parameters.

    Parameters
    ----------
    params : dict[str, float]
        A dictionary of parameters to update.

    keys : Iterable[str]
        Keys to set.

    values : Iterable[float]
        Values to set.

    Returns
    -------
    Optional[dict[str, float]]
        A fully defined parameters dictionary if `params` is sufficient and consistent
        updated after setting new parameter values. Otherwise, None.
    """
    assert len(keys) == len(values), "Length of keys to update in params must equal length of values."
    for key, val in zip(keys, values):
        params[key] = val
    return get_params_from_dict(params)


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
