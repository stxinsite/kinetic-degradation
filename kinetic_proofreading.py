import numpy as np
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt

"""
Functions in KINETIC RATES section must be provided `params` dictionary with the following fields:

alpha: cooperativity
Kd_T_binary: equilibrium dissociation constant of T in binary complex
kon_T_binary: kon of BPD + T -> BPD-T
koff_T_binary: koff of BPD-T -> BPD + T
Kd_T_ternary: equilibrium dissociation constant of T in ternary complex
kon_T_ternary: kon of BPD-E3 + T -> T-BPD-E3
koff_T_ternary: koff of T-BPD-E3 -> BPD-E3 + T
Kd_E3_binary: equilibrium dissociation constant of E3 in binary complex
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
    return - params['PS_cell'] * params['num_cells'] * \
           ( (params['fu_ec'] * BPD_ec / params['Vec']) - (params['fu_ic'] * BPD_ic / params['Vic']) )

def dBPD_icdt(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3):
    return params['PS_cell'] * ((params['fu_ec'] * BPD_ec / params['Vec']) - (params['fu_ic'] * BPD_ic / params['Vic'])) - \
           params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic'] + \
           params['koff_T_binary'] * BPD_T - \
           params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic'] + \
           params['koff_E3_binary'] * BPD_E3 + \
           params['kdeg_T'] * BPD_T

def dTargetdt(params, BPD_ic, T, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    return params['kprod_T'] - params['kdeg_T'] * T - \
           params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic'] + \
           params['koff_T_binary'] * BPD_T - \
           params['kon_T_ternary'] * BPD_E3 * T / params['Vic'] + \
           params['koff_T_ternary'] * (Ternary + np.sum(Ternary_Ubs))

def dE3dt(params, BPD_ic, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    return - params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic'] + \
           params['koff_E3_binary'] * BPD_E3 - \
           params['kon_E3_ternary'] * BPD_T * E3 / params['Vic'] + \
           params['koff_E3_ternary'] * (Ternary + np.sum(Ternary_Ubs))

def dBPD_Tdt(params, BPD_ic, T, E3, BPD_T, Ternary, *Ternary_Ubs):
    return params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic'] - \
           params['koff_T_binary'] * BPD_T - \
           params['kon_E3_ternary'] * BPD_T * E3 / params['Vic'] + \
           params['koff_E3_ternary'] * (Ternary + np.sum(Ternary_Ubs)) - \
           params['kdeg_T'] * BPD_T

def dBPD_E3dt(params, BPD_ic, T, E3, BPD_E3, Ternary, *Ternary_Ubs):
    return params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic'] - \
           params['koff_E3_binary'] * BPD_E3 - \
           params['kon_T_ternary'] * BPD_E3 * T / params['Vic'] + \
           (params['koff_T_ternary'] + params['kdeg_T']) * (Ternary + np.sum(Ternary_Ubs)) + \
           params['ktransit_UPS'] * (Ternary_Ubs[-1] if params['n'] > 0 else 0)

def dTernarydt(params, T, E3, BPD_T, BPD_E3, Ternary):
    return params['kon_T_ternary'] * BPD_E3 * T / params['Vic'] + \
           params['kon_E3_ternary'] * BPD_T * E3 / params['Vic'] - \
           (params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['ktransit_UPS']) * Ternary

def dTernary_Ubdt(Ternary_Ub_consec, params):
    return params['ktransit_UPS'] * Ternary_Ub_consec[0] - \
           (params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['ktransit_UPS']) * Ternary_Ub_consec[1]

def kinetic_rates(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    BPD_ec_rate = dBPD_ecdt(params, BPD_ec, BPD_ic)
    BPD_ic_rate = dBPD_icdt(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3)
    T_rate = dTargetdt(params, BPD_ic, T, BPD_T, BPD_E3, Ternary, *Ternary_Ubs)
    E3_rate = dE3dt(params, BPD_ic, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs)
    BPD_T_rate = dBPD_Tdt(params, BPD_ic, T, E3, BPD_T, Ternary, *Ternary_Ubs)
    BPD_E3_rate = dBPD_E3dt(params, BPD_ic, T, E3, BPD_E3, Ternary, *Ternary_Ubs)
    Ternary_rate = dTernarydt(params, T, E3, BPD_T, BPD_E3, Ternary)
    if n > 0:  # if there is at least one ubiquitination step
        Ternary_all = np.insert(np.array(Ternary_Ubs), 0, Ternary)  # prepend Ternary to Ternary_Ubs
        Ternary_pairs = np.lib.stride_tricks.sliding_window_view(Ternary_all, 2)  # create array of consecutive pairs
        Ternary_Ubs_rates = np.apply_along_axis(dTernary_Ubdt, 1, Ternary_pairs, params = params).tolist()  # apply dTernary_Ubdt() to each pair
    else:
        Ternary_Ubs_rates = []  # no rates for ubiquitinated Ternary complexes if there are none

    all_rates = np.array(
        [BPD_ec_rate, BPD_ic_rate, T_rate, E3_rate, BPD_T_rate, BPD_E3_rate, Ternary_rate] + Ternary_Ubs_rates
    )
    return all_rates

def jac_kinetic_rates(params, BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    """
    Jacobian of rates with respect to variables.
    dBPD_ecdtdy, dBPD_icdtdy, ..., dTernarydtdy: list < 7 + n >
    dTernary_Ubdtdy_all: list-of-lists < n, 7 + n >
    """
    dBPD_ecdtdy = (
        [
            -params['PS_cell'] * params['num_cells'] * params['fu_ec'] / Vec,
            params['PS_cell'] * params['num_cells'] * params['fu_ic'] / params['Vic']
        ]
        + [0] * (5 + params['n'])  # does not depend on T, E3, BPD_T, BPD_E3, Ternary, Ternary_Ubs
    )
    dBPD_icdtdy = (
        [
            params['PS_cell'] * params['fu_ec'] / Vec,
            -params['PS_cell'] * params['fu_ic'] / params['Vic'] - params['kon_T_binary'] * params['fu_ic'] * T / params['Vic'] - params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
            -params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            -params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            params['koff_T_binary'] + params['kdeg_T'],
            params['koff_E3_binary'],
            0
        ]
        + [0] * params['n']  # does not depend on Ternary_Ubs
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
        ]
        + [params['koff_T_ternary']] * params['n']
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
        + [params['koff_E3_ternary']] * params['n']
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
        + [params['koff_E3_ternary']] * params['n']
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
        ]
        + [params['koff_T_ternary'] + params['kdeg_T']] * (params['n'] - 1)  # w.r.t. Ternary_Ub_1, ..., Ternary_Ub_<n-1>. If n == 0, then becomes empty list.
        + [params['koff_T_ternary'] + params['kdeg_T'] + params['ktransit_UPS']] * (1 if params['n'] > 0 else 0)  # w.r.t. Ternary_Ub_n
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
        ]
        + [0] * params['n']  # does not depend on Ternary_Ubs
    )
    dTernary_Ubdtdy_all = []  # initialize empty list for dTernary_Ubdt / dy
    if params['n'] > 0:  # if there are ubiquitinated Ternary complexes
        for i in range(params['n']):  # for each Ternary complex transit compartment i
            dTernary_Ub_idtdy = [0] * (7 + params['n'])  # initalize zeros list for dTernary_Ub_i / dt / dy
            dTernary_Ub_idtdy[6 + i] = params['ktransit_UPS']  # dTernary_Ub_i / dt / d[Ternary_Ub_<i-1>]
            dTernary_Ub_idtdy[7 + i] = -(params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['ktransit_UPS'])  # dTernary_Ub_i / dt / d[Ternary_Ub_i]
            dTernary_Ubdtdy_all.append(dTernary_Ub_idtdy)  # append dTernary_Ub_i / dt / dy list to list-of-lists

    all_jacs = np.array(  # (7 + n) x (7 + n) array
        [dBPD_ecdtdy, dBPD_icdtdy, dTargetdtdy, dE3dtdy, dBPD_Tdtdy, dBPD_E3dtdy, dTernarydtdy]  # 7 x (7 + n) list-of-lists
        + dTernary_Ubdtdy_all  # n x (7 + n) list-of-lists
    )
    return all_jacs

def calc_concentrations(times, y0, params, max_step = np.inf, rtol = 1e-3, atol = 1e-6):
    def rates(t, y, params):
        return kinetic_rates(params, *y)

    def jac_rates(t, y, params):
        return jac_kinetic_rates(params, *y)

    tmin = np.min(times)
    tmax = np.max(times)
    # dtimes = times[1:] - times[:-1]  # intervals between times

    results = integrate.solve_ivp(rates, (tmin, tmax), y0,
                                  args = (params, ),
                                  method = 'BDF',
                                  t_eval = times,
                                  jac = jac_rates,
                                  max_step = max_step,
                                  rtol = rtol,
                                  atol = atol
                                  )

    return results

"""
Manipulate results from solving system of kinetic rates.
"""
def dataframe_concentrations(solve_ivp_result):
    """
    Creates pandas.DataFrame from result object of scipy.integrate.solve_ivp()
    """
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
                                ]
                                + ['Ternary_Ub_' + str(i) for i in range(1, n + 1)]  # empty list if n == 0
                              )
                             )
    results_df['t'] = solve_ivp_result.t
    return results_df

def plot_concentrations(results_df):
    """
    Plots columns of result from *.dataframe_concentrations() against time column `t`.

    results_df <pandas.DataFrame>: dataframe with `t` column and species columns
    """
    plt.rcParams["axes.labelsize"] = 20
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
