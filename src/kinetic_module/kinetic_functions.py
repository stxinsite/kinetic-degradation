import numpy as np
import pandas as pd
import scipy.integrate as integrate
from scipy.optimize import root
from matplotlib import pyplot as plt

"""
To do:
- how to assert plausibility of solve_ivp() results
    - all values should be non-negative
    - but because of numerical solver some values can be negative
        - these values must be close to zero
    - current implementation: recursively halve max_step until all results are non-negative
"""


"""
Functions in KINETIC RATES section must be provided `params` dictionary with the following fields:

alpha: cooperativity
Kd_T_binary: equilibrium dissociation constant of BPD-T binary complex
kon_T_binary: kon of BPD + T -> BPD-T
koff_T_binary: koff of BPD-T -> BPD + T
Kd_T_ternary: equilibrium dissociation constant of T in ternary complex
kon_T_ternary: kon of BPD-E3 + T -> T-BPD-E3
koff_T_ternary: koff of T-BPD-E3 -> BPD-E3 + T
Kd_E3_binary: equilibrium dissociation constant of BPD-E3 binary complex
kon_E3_binary: kon of BPD + E3 -> BPD-E3
koff_E3_binary: koff of BPD-E3 -> BPD + E3
Kd_E3_ternary: equilibrium dissociation constant of E3 in ternary complex
kon_E3_ternary: kon of BPD-T + E3 -> T-BPD-E3
koff_E3_ternary: koff of T-BPD-E3 -> BPD-T + E3
n: number of ubiquitination steps before degradation
MTT_deg: mean transit time of degradation
ktransit_UPS: transit rate
fu_ec: fraction unbound extracellular BPD
fu_ic: fraction unbound intracellular BPD
PS_cell: permeability-surface area product
kprod_T: baseline target protein production rate
kdeg_T: baseline target protein degradation rate
Conc_T_base: baseline target protein concentration
Conc_E3_base: baseline E3 concentration
num_cells: number of cells in system
Vic: intracellular volume
Vec: extracellular volume
"""

"""KINETIC RATES"""
def dBPD_ecdt(params, BPD_ec, BPD_ic):
    """
    Calculates dBPD_ec / dt.
    """
    return -params['PS_cell'] * params['num_cells'] * \
           ( (params['fu_ec'] * BPD_ec / params['Vec']) - (params['fu_ic'] * BPD_ic / params['Vic']) )

def dBPD_icdt(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3):
    """
    Calculates dBPD_ic / dt.
    """
    return params['PS_cell'] * ((params['fu_ec'] * BPD_ec / params['Vec']) - (params['fu_ic'] * BPD_ic / params['Vic'])) - \
           params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic'] + \
           params['koff_T_binary'] * BPD_T - \
           params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic'] + \
           params['koff_E3_binary'] * BPD_E3 + \
           params['kdeg_T'] * BPD_T

def dTargetdt(params, BPD_ic, T, BPD_T, BPD_E3, Ternary, Ternary_Ubs):
    """
    Calculates dTarget / dt.
    """
    return params['kprod_T'] - params['kdeg_T'] * T - \
           params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic'] + \
           params['koff_T_binary'] * BPD_T - \
           params['kon_T_ternary'] * BPD_E3 * T / params['Vic'] + \
           params['koff_T_ternary'] * (Ternary + np.sum(Ternary_Ubs))

def dE3dt(params, BPD_ic, E3, BPD_T, BPD_E3, Ternary, Ternary_Ubs):
    """
    Calculates dE3 / dt.
    """
    return - params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic'] + \
           params['koff_E3_binary'] * BPD_E3 - \
           params['kon_E3_ternary'] * BPD_T * E3 / params['Vic'] + \
           params['koff_E3_ternary'] * (Ternary + np.sum(Ternary_Ubs))

def dBPD_Tdt(params, BPD_ic, T, E3, BPD_T, Ternary, Ternary_Ubs):
    """
    Calculates dBPD_T / dt.
    """
    return params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic'] - \
           params['koff_T_binary'] * BPD_T - \
           params['kon_E3_ternary'] * BPD_T * E3 / params['Vic'] + \
           params['koff_E3_ternary'] * (Ternary + np.sum(Ternary_Ubs)) - \
           params['kdeg_T'] * BPD_T

def dBPD_E3dt(params, BPD_ic, T, E3, BPD_E3, Ternary, Ternary_Ubs):
    """
    Calculates dBPD_E3 / dt.
    """
    return params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic'] - \
           params['koff_E3_binary'] * BPD_E3 - \
           params['kon_T_ternary'] * BPD_E3 * T / params['Vic'] + \
           (params['koff_T_ternary'] + params['kdeg_T']) * (Ternary + np.sum(Ternary_Ubs)) + \
           params['ktransit_UPS'] * (Ternary_Ubs[-1] if params['n'] > 0 else 0)

def dTernarydt(params, T, E3, BPD_T, BPD_E3, Ternary):
    """
    Calculates dTernary / dt.
    """
    return params['kon_T_ternary'] * BPD_E3 * T / params['Vic'] + \
           params['kon_E3_ternary'] * BPD_T * E3 / params['Vic'] - \
           (params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['ktransit_UPS']) * Ternary

def dTernary_Ubdt(Ternary_Ub_consec_pair, params):
    """
    Calculates dTernary_Ub_i / dt.

    Args:

    Ternary_Ub_consec_pair: float; an array_like of length 2 containing Ternary_Ub_<i-1> followed by Ternary_Ub_i

    params: dict; contains kinetic parameters
    """
    return params['ktransit_UPS'] * Ternary_Ub_consec_pair[0] - \
           (params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['ktransit_UPS']) * Ternary_Ub_consec_pair[1]

def kinetic_rates(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    """
    Calculates rates of change for the species in PROTAC-mediated target protein degradation system.

    Args:

    params: dict; contains parameters for rate equations.

    BPD_ec: float; amount of unbound extracellular Bispecific Protein Degrader.

    BPD_ic: float; amount of unbound intracellular Bispecific Protein Degrader.

    T: float; amount of unbound Target protein.

    E3: float; amount of unbound E3 ligase.

    BPD_T: float; amount of BPD-T binary complex.

    BPD_E3: float; amount of BPD-E3 binary complex.

    Ternary: float; amount of T-BPD-E3 ternary complex.

    Ternary_Ubs: float; amounts of ubiquitinated ternary complex in increasing order of length of ubiquitin chain.
    """
    BPD_ec_rate = dBPD_ecdt(params, BPD_ec, BPD_ic)
    BPD_ic_rate = dBPD_icdt(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3)
    T_rate = dTargetdt(params, BPD_ic, T, BPD_T, BPD_E3, Ternary, Ternary_Ubs)
    E3_rate = dE3dt(params, BPD_ic, E3, BPD_T, BPD_E3, Ternary, Ternary_Ubs)
    BPD_T_rate = dBPD_Tdt(params, BPD_ic, T, E3, BPD_T, Ternary, Ternary_Ubs)
    BPD_E3_rate = dBPD_E3dt(params, BPD_ic, T, E3, BPD_E3, Ternary, Ternary_Ubs)
    Ternary_rate = dTernarydt(params, T, E3, BPD_T, BPD_E3, Ternary)
    if params['n'] > 0:  # if there is at least one ubiquitination step
        Ternary_all = np.insert(np.array(Ternary_Ubs), 0, Ternary)  # prepend Ternary to Ternary_Ubs
        Ternary_pairs = np.lib.stride_tricks.sliding_window_view(Ternary_all, 2)  # create array of sliding consecutive pairs
        Ternary_Ubs_rates = np.apply_along_axis(dTernary_Ubdt, 1, Ternary_pairs, params = params).tolist()  # apply dTernary_Ubdt() to each pair
    else:
        Ternary_Ubs_rates = []  # else no ubiquitinated Ternary complexes

    all_rates = np.array(
        [BPD_ec_rate, BPD_ic_rate, T_rate, E3_rate, BPD_T_rate, BPD_E3_rate, Ternary_rate] + Ternary_Ubs_rates  # list concatenation
    )

    return all_rates

def jac_kinetic_rates(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    """
    Calculates M x M Jacobian of rates with respect to M species.
    Identical call signature as kinetic_rates().

    dBPD_ecdtdy, dBPD_icdtdy, ..., dTernarydtdy: list, shape(7 + n)
    dTernary_Ubdtdy_all: list of lists, shape(n, 7 + n)
    """
    dBPD_ecdtdy = (
        [
            -params['PS_cell'] * params['num_cells'] * params['fu_ec'] / params['Vec'],
            params['PS_cell'] * params['num_cells'] * params['fu_ic'] / params['Vic']
        ] +
        [0] * (5 + params['n'])  # dBPD_ec/dt does not depend on T, E3, BPD_T, BPD_E3, Ternary, Ternary_Ubs
    )
    dBPD_icdtdy = (
        [
            params['PS_cell'] * params['fu_ec'] / params['Vec'],
            -params['PS_cell'] * params['fu_ic'] / params['Vic'] - params['kon_T_binary'] * params['fu_ic'] * T / params['Vic'] - params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
            -params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            -params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            params['koff_T_binary'] + params['kdeg_T'],
            params['koff_E3_binary'],
            0
        ] +
        [0] * params['n']  # dBPD_ic/dt does not depend on Ternary_Ubs
    )
    dTargetdtdy = (
        [
            0,
            -params['kon_T_binary'] * params['fu_ic'] * T / params['Vic'],
            -params['kdeg_T'] - params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'] - params['kon_T_ternary'] * BPD_E3 / params['Vic'],
            0,
            params['koff_T_binary'],
            -params['kon_T_ternary'] * T / params['Vic'],
            params['koff_T_ternary']
        ] +
        [ params['koff_T_ternary'] ] * params['n']
    )
    dE3dtdy = (
        [
            0,
            -params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
            0,
            -params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'] - params['kon_E3_ternary'] * BPD_T / params['Vic'],
            -params['kon_E3_ternary'] * E3 / params['Vic'],
            params['koff_E3_binary'],
            params['koff_E3_ternary']
        ]
        + [ params['koff_E3_ternary'] ] * params['n']
    )
    dBPD_Tdtdy = (
        [
            0,
            params['kon_T_binary'] * params['fu_ic'] * T / params['Vic'],
            params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            -params['kon_E3_ternary'] * BPD_T / params['Vic'],
            -params['koff_T_binary'] - params['kon_E3_ternary'] * E3 / params['Vic'] - params['kdeg_T'],
            0,
            params['koff_E3_ternary']
        ]
        + [ params['koff_E3_ternary'] ] * params['n']
    )
    dBPD_E3dtdy = (
        [
            0,
            params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
            -params['kon_T_ternary'] * BPD_E3 / params['Vic'],
            params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            0,
            -params['koff_E3_binary'] - params['kon_T_ternary'] * T / params['Vic'],
            params['koff_T_ternary'] + params['kdeg_T']
        ] +
        [ params['koff_T_ternary'] + params['kdeg_T'] ] * (params['n'] - 1) +  # if n = 0, empty list.
        [ params['koff_T_ternary'] + params['kdeg_T'] + params['ktransit_UPS'] ] * (1 if params['n'] > 0 else 0)  # with respect to Ternary_Ub_n
    )
    dTernarydtdy = (
        [
            0,
            0,
            params['kon_T_ternary'] * BPD_E3 / params['Vic'],
            params['kon_E3_ternary'] * BPD_T / params['Vic'],
            params['kon_E3_ternary'] * E3 / params['Vic'],
            params['kon_T_ternary'] * T / params['Vic'],
            -(params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['ktransit_UPS'])
        ] +
        [0] * params['n']  # dTernary/dt does not depend on Ternary_Ubs
    )
    dTernary_Ubdtdy_all = []  # initialize empty list for (dTernary_Ub/dt) / dy
    if params['n'] > 0:  # if there are ubiquitination steps
        for i in range(params['n']):  # for each Ternary complex transit step i
            dTernary_Ub_idtdy = [0] * (7 + params['n'])  # initalize list of zeros for (dTernary_Ub_i/dt) / dy
            dTernary_Ub_idtdy[6 + i] = params['ktransit_UPS']  # (dTernary_Ub_i/dt) / dTernary_Ub_<i-1>
            dTernary_Ub_idtdy[7 + i] = -(params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['ktransit_UPS'])  # (dTernary_Ub_i/dt) / dTernary_Ub_i
            dTernary_Ubdtdy_all.append(dTernary_Ub_idtdy)  # append (dTernary_Ub_i/dt) / dy

    # (7 + n) x (7 + n) array
    all_jacs = np.array(
        [dBPD_ecdtdy, dBPD_icdtdy, dTargetdtdy, dE3dtdy, dBPD_Tdtdy, dBPD_E3dtdy, dTernarydtdy] +  # 7 x (7 + n) list of lists
        dTernary_Ubdtdy_all  # n x (7 + n) list of lists
    )

    return all_jacs

def calc_concentrations(params, times, y0, max_step = np.inf, T_total_baseline = None, T_total_steady_state = None):
    """
    Solve the initial value problem for the amounts of species (unbound, binary complexes, ternary complexes) in
    PROTAC-induced target protein degradation via the ubiquitin-proteasome system.

    Args:
        params: dict; the parameters for the rate equations.
        times: array_like; the time points at which the amounts are calculated.
        y0: array_like; the initial values of all species in system in the following order:
            [BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, Ternary_Ub_1, ..., Ternary_Ub_n].
        max_step: float; maximum allowed step size for solver.
    """
    def rates(t, y, params, T_total_baseline, T_total_steady_state):
        """
        Returns ODEs evaluated at *y with params.

        Must have call signature func(t, y, *args).
        """
        return kinetic_rates(params, *y)

    def jac_rates(t, y, params, T_total_baseline, T_total_steady_state):
        """
        Returns Jacobian of ODEs with respect to *y evaluated at *y.

        Must have identical call signature as rates().
        """
        return jac_kinetic_rates(params, *y)

    def t_half_event(t, y, params, T_total_baseline, T_total_steady_state):
        C = 0.5 * (T_total_baseline - T_total_steady_state)
        T_total_at_t = np.sum(np.concatenate((y[[2,4]], y[6:])))
        return T_total_at_t - C

    t_half_event.terminal = True  # if t_half_event = 0, solver will terminate
    # tmin = np.min(times)
    tmax = np.max(times)
    # dtimes = times[1:] - times[:-1]  # intervals between times

    results = integrate.solve_ivp(rates, (0, tmax), y0,
                                  method = 'BDF',
                                  t_eval = times,
                                  args = (params, T_total_baseline, T_total_steady_state),
                                  max_step = max_step,
                                  jac = jac_rates,
                                  events = t_half_event if (T_total_baseline is not None) and (T_total_steady_state is not None) else None
                                  )
    if not np.all(results.y >= 0):
        print(params)
        print(results.y)
        max_step = max_step / 2
        return calc_concentrations(params, times, y0, max_step, T_total_baseline, T_total_steady_state)  # halve the max_step
    # assert np.all(results.y >= 0), f"Results from solve_ivp() at initial values {y0} are not all non-negative."
    return results

def wrap_kinetic_rates(arr, params):
    return kinetic_rates(params, *arr)

def wrap_jac_kinetic_rates(arr, params):
    return jac_kinetic_rates(params, *arr)

def solve_steady_state(initial_guess, params):
    """
    Solve system steady state, the amounts of species for which all kinetic rates
    equal 0.

    Args:
        initial_guess: array; initial guess for solution
        params: dict; model config

    Returns:
        array; amounts of species at steady state
    """
    methods = ['hybr', 'lm']
    for m in methods:
        if m == 'hybr':
            # options = {'maxfev': 5000}
            options = {'xtol': 1.49012e-9, 'maxfev': 5000}
        elif m == 'lm':
            options = {'xtol': 1.49012e-9, 'maxiter': 5000}
            # options = {'xtol': 1.49012e-10, 'ftol': 1.49012e-09, 'maxiter': 10000}
        else:
            options = None
        roots = root(
            wrap_kinetic_rates,
            initial_guess,
            jac = wrap_jac_kinetic_rates,
            args = (params,),
            method = m,
            options = options
            )
        if roots.success and np.all(roots.x >= 0):
            break

    if (not roots.success) or np.any(roots.x < 0):
        print(roots.message)
        # return None

    return roots.x

def calc_Dmax(params, times, y0, initial_guess = None):
    """
    Calculate Dmax
    Returns:
        float; maximum percent target protein degradation
    """
    if initial_guess is None:
        print("Solving IVP because no initial guess.")
        result = calc_concentrations(params, times, y0, max_step = 0.001)
        initial_guess = result.y[:,-1]  # system state at the last time point
    steady_state = solve_steady_state(initial_guess, params)
    T_total_steady_state = np.sum(np.concatenate((steady_state[[2,4]], steady_state[6:])))
    T_total_baseline = np.sum(np.concatenate((y0[[2,4]], y0[6:])))
    Dmax = 1 - T_total_steady_state / T_total_baseline
    return Dmax

def calc_t_half(params, times, y0):
    """
    Find time t at which total Target amount reaches halfway between baseline and
    steady state.
    """
    result = calc_concentrations(params, times, y0, max_step = 0.001)
    init_guess = result.y[:,-1]  # system state at the last time point
    steady_state = solve_steady_state(init_guess, params)
    T_total_steady_state = np.sum(np.concatenate((steady_state[[2,4]], steady_state[6:])))
    T_total_baseline = np.sum(np.concatenate((y0[[2,4]], y0[6:])))

    result_with_events = calc_concentrations(params, times, y0, max_step = 0.001,
        T_total_baseline = T_total_baseline, T_total_steady_state = T_total_steady_state)

    t_half = result_with_events.t_events[0][0]  # first event type, first time point
    return t_half

"""
RESULT MANIPULATION AND VISUALIZATION
"""
def dataframe_concentrations(solve_ivp_result):
    """
    Create pandas.DataFrame from result object of scipy.integrate.solve_ivp()
    where [i,j] is amount of species j at time i.

    Args:
        solve_ivp_result: OptimizeResult; result returned by scipy.integrate.solve_ivp().
    """
    num_species = solve_ivp_result.y.shape[0]  # total number of species in system
    num_Ub_compartments = num_species - 7  # ubiquitin compartments occur after the first 7 species. See y0 in calc_concentrations().

    results_df = pd.DataFrame(solve_ivp_result.y.T,
                              columns = (
                                [
                                    'BPD_ec',
                                    'BPD_ic',
                                    'T',
                                    'E3',
                                    'BPD_T',
                                    'BPD_E3',
                                    'Ternary'
                                ] +
                                [ 'Ternary_Ub_' + str(i) for i in range(1, num_Ub_compartments + 1) ]  # empty list if no ubiquitin compartments
                              )
                             )
    results_df['t'] = solve_ivp_result.t  # add column `t` for time points

    return results_df

def plot_concentrations(results_df):
    """
    Plots columns of species amounts from dataframe_concentrations() against time column `t`.

    Args:

    results_df: pandas.DataFrame; contains column `t` for time and other columns named for species
    """
    ax = results_df.plot(x='t',
                         xlabel = 'Time (hours)',
                         ylabel = 'Amount (uM)',
                         kind='bar',
                         stacked=True,
                         logy = False,
                         title='Amounts of species in kinetic model',
                         figsize = (12, 8),
                         fontsize = 20
                         )
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()
