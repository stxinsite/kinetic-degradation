import scipy.integrate as integrate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""GLOBAL VARIABLES"""
# BPD PARAMETERS
alpha = 1.

Kd_T_binary = 0.091
kon_T_binary = 3600.
koff_T_binary = kon_T_binary * Kd_T_binary

Kd_T_ternary = Kd_T_binary / alpha
koff_T_ternary = 327.6
kon_T_ternary = koff_T_ternary / Kd_T_ternary

Kd_E3_binary = 3.1
kon_E3_binary = 3600.
koff_E3_binary = kon_E3_binary * Kd_E3_binary

Kd_E3_ternary = Kd_E3_binary / alpha
koff_E3_ternary = 11160.
kon_E3_ternary = koff_E3_ternary / Kd_E3_ternary

n = 0  # 3
MTT_deg = 0.0015
ktransit_UPS = 0  # (n + 1) / MTT_deg
fu_c = np.nan
fu_ec = 1.
fu_ic = 1.
F = np.nan
ka = np.nan
CL = np.nan
Vc = np.nan
Q = np.nan
Vp = np.nan
PS_cell = 0  # 1e-12
PSV_tissue = np.nan
MW_BPD = 947.

# PHYSIOLOGICAL SYSTEM PARAMETERS
kdeg_T = 0  # 0.058
Conc_T_base = 0.2
Conc_E3_base = 0.5
num_cells = 5e3
Vic = 1e-4
Vec = 2e-4
kprod_T = Conc_T_base * Vic * kdeg_T
BW = np.nan

"""INITIAL VALUES"""
BPD_ev = 0
BPD_c = 0
BPD_p = 0
BPD_ec = 0  # nM * Vec / 1000
BPD_ic = 1000 * Vic / 1000  # nM * Vic / 1000
T = Conc_T_base * Vic
E3 = Conc_E3_base * Vic
BPD_T = 0
BPD_E3 = 0
Ternary = 0
Ternary_Ubs = np.zeros(n)  # where i = 0 is un-ubiquitinated Ternary

"""KINETIC RATES"""
def dBPD_ecdt(BPD_ec, BPD_ic):
    return -PS_cell * num_cells * ((fu_ec * BPD_ec / Vec) - (fu_ic * BPD_ic / Vic))

def dBPD_icdt(BPD_ic, T, E3, BPD_T, BPD_E3):
    return PS_cell * ((fu_ec * BPD_ec / Vec) - (fu_ic * BPD_ic / Vic)) - \
           kon_T_binary * fu_ic * BPD_ic * T / Vic + \
           koff_T_binary * BPD_T - \
           kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \
           koff_E3_binary * BPD_E3 + \
           kdeg_T * BPD_T

def dTargetdt(BPD_ic, T, BPD_T, BPD_E3, Ternary):
    return kprod_T - kdeg_T * T - \
           kon_T_binary * fu_ic * BPD_ic * T / Vic + \
           koff_T_binary * BPD_T - \
           kon_T_ternary * BPD_E3 * T / Vic + \
           koff_T_ternary * (Ternary + np.sum(Ternary_Ubs))

def dE3dt(BPD_ic, E3, BPD_T, BPD_E3, Ternary):
    return -kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \
            koff_E3_binary * BPD_E3 - \
            kon_E3_ternary * BPD_T * E3 / Vic + \
            koff_E3_ternary * (Ternary + np.sum(Ternary_Ubs))

def dBPD_Tdt(BPD_ic, T, E3, BPD_T, Ternary):
    return kon_T_binary * fu_ic * BPD_ic * T / Vic - \
           koff_T_binary * BPD_T - \
           kon_E3_ternary * BPD_T * E3 / Vic + \
           koff_E3_ternary * (Ternary + np.sum(Ternary_Ubs)) - \
           kdeg_T * BPD_T

def dBPD_E3dt(BPD_ic, T, E3, BPD_E3, Ternary):
    return kon_E3_binary * fu_ic * BPD_ic * E3 / Vic - \
           koff_E3_binary * BPD_E3 - \
           kon_T_ternary * BPD_E3 * T / Vic + \
           (koff_T_ternary + kdeg_T) * (Ternary + np.sum(Ternary_Ubs)) + \
           ktransit_UPS * Ternary_Ubs[-1]

def dTernarydt(T, E3, BPD_T, BPD_E3, Ternary):
    return kon_T_ternary * BPD_E3 * T / Vic + \
           kon_E3_ternary * BPD_T * E3 / Vic - \
           (kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS) * Ternary

def dTernary_Ubdt(Ternary_Ub_pair):
    return ktransit_UPS * Ternary_Ub_pair[0] - \
           (kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS) * Ternary_Ub_pair[1]

def rates_ternary_formation(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    BPD_ec_rate = dBPD_ecdt(BPD_ec, BPD_ic)
    BPD_ic_rate = dBPD_icdt(BPD_ic, T, E3, BPD_T, BPD_E3)
    T_rate = dTargetdt(BPD_ic, T, BPD_T, BPD_E3, Ternary)
    E3_rate = dE3dt(BPD_ic, E3, BPD_T, BPD_E3, Ternary)
    BPD_T_rate = dBPD_Tdt(BPD_ic, T, E3, BPD_T, Ternary)
    BPD_E3_rate = dBPD_E3dt(BPD_ic, T, E3, BPD_E3, Ternary)
    Ternary_rate = dTernarydt(T, E3, BPD_T, BPD_E3, Ternary)
    if len(Ternary_Ubs):  # if there is at least one ubiquitination step
        Ternary_all = np.insert(np.array(Ternary_Ubs), 0, Ternary)  # prepend Ternary to Ternary_Ubs
        Ternary_pairs = np.lib.stride_tricks.sliding_window_view(Ternary_all, 2)  # create sliding window of pairs
        Ternary_Ubs_rates = np.apply_along_axis(dTernary_Ubdt, 1, Ternary_pairs)  # apply dTernary_Ubdt to each pair
    else:
        Ternary_Ubs_rates = np.empty(0)

    ternary_formation_rates = np.array([BPD_ic_rate, T_rate, E3_rate, BPD_T_rate, BPD_E3_rate, Ternary_rate])
    all_rates = np.concatenate((ternary_formation_rates, Ternary_Ubs_rates))
    return all_rates

def jac_rates_ternary_formation(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    """
    df / dy = [ df/dBPD_ic, df/dT, df/dE3, df/dBPD_T, df/dBPD_E3, df/dTernary ]
    """
    dBPD_ecdtdy = np.array([
        -PS_cell * num_cells * fu_ec / Vec,
        PS_cell * num_cells * fu_ic / Vic
    ] + [0] * (5 + n)
    )
    dBPD_icdtdy = np.array([

        -PS_cell * fu_ic / Vic - kon_T_binary * fu_ic * T / Vic - kon_E3_binary * fu_ic * E3 / Vic,
        -kon_T_binary * fu_ic * BPD_ic / Vic,
        -kon_E3_binary * fu_ic * BPD_ic / Vic,
        koff_T_binary + kdeg_T,
        koff_E3_binary,
        0
    ])
    dTargetdtdy = np.array([
        -kon_T_binary * fu_ic * T / Vic,
        -kdeg_T - kon_T_binary * fu_ic * BPD_ic / Vic - kon_T_ternary * BPD_E3 / Vic,
        0,
        koff_T_binary,
        -kon_T_ternary * T / Vic,
        koff_T_ternary
    ])
    dE3dtdy = np.array([
        -kon_E3_binary * fu_ic * E3 / Vic,
        0,
        -kon_E3_binary * fu_ic * BPD_ic / Vic - kon_E3_ternary * BPD_T / Vic,
        -kon_E3_ternary * E3 / Vic,
        koff_E3_binary,
        koff_E3_ternary
    ])
    dBPD_Tdtdy = np.array([
        kon_T_binary * fu_ic * T / Vic,
        kon_T_binary * fu_ic * BPD_ic / Vic,
        -kon_E3_ternary * BPD_T / Vic,
        -koff_T_binary - kon_E3_ternary * E3 / Vic - kdeg_T,
        0,
        koff_E3_ternary
    ])
    dBPD_E3dtdy = np.array([
        kon_E3_binary * fu_ic * E3 / Vic,
        -kon_T_ternary * BPD_E3 / Vic,
        kon_E3_binary * fu_ic * BPD_ic / Vic,
        0,
        -koff_E3_binary - kon_T_ternary * T / Vic,
        koff_T_ternary + kdeg_T
    ])
    dTernarydtdy = np.array([
        0,
        kon_T_ternary * BPD_E3 / Vic,
        kon_E3_ternary * BPD_T / Vic,
        kon_E3_ternary * E3 / Vic,
        kon_T_ternary * T / Vic,
        -(kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS)
    ])

    return np.array([dBPD_icdtdy, dTargetdtdy, dE3dtdy, dBPD_Tdtdy, dBPD_E3dtdy, dTernarydtdy])

def calc_concentrations(times, y0):
    def rates(t, y):
        return rates_ternary_formation(*y)

    def jac_rates(t, y):
        return jac_rates_ternary_formation(*y)

    tmin = np.min(times)
    tmax = np.max(times)
    results = integrate.solve_ivp(rates, (tmin, tmax), y0,
                                  method = 'BDF',
                                  t_eval = times,
                                  jac = jac_rates
                                  )

    return results

def plot_concentrations(times, ytimes):
    results_df = pd.DataFrame(ytimes.T, columns = ['BPD_ic', 'T', 'E3', 'BPD_T', 'BPD_E3', 'Ternary'])
    results_df['t'] = times

    # plt.rcParams["figure.autolayout"] = True
    ax = results_df.plot(x='t',
                         xlabel = 'Time (hours)',
                         ylabel = 'Amount (uM)',
                         kind='bar',
                         stacked=True,
                         logy = False,
                         title='Amounts of species in ternary complex kinetic model',
                         figsize = (10, 5)
                         )
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()
    return results_df

"""SIMULATIONS"""
# species amounts at time = 0
y0 = np.array([BPD_ic, T, E3, BPD_T, BPD_E3, Ternary])
# time steps
t = np.arange(start = 0, stop = 168 + 1, step = 24)
# t = np.array([0, 0.5, 1])

results = calc_concentrations(times = t, y0 = y0)
results.success

results_df = plot_concentrations(t, results.y)

BPD_total = results_df[['BPD_ic', 'BPD_T', 'BPD_E3', 'Ternary']].sum(axis = 1)
np.allclose(BPD_total, BPD_total[0])

T_total = results_df[['T', 'BPD_T', 'Ternary']].sum(axis = 1)
np.allclose(T_total, T_total[0])

E3_total = results_df[['E3', 'BPD_E3', 'Ternary']].sum(axis = 1)
np.allclose(E3_total, E3_total[0])

"""BARTLETT SUPPLEMENTARY FIGURE 1 (a)"""
Conc_BPD_ic_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)
Ternary_formation_arr = np.empty((len(Conc_BPD_ic_arr),2))

for count, conc in enumerate(Conc_BPD_ic_arr):
    y0 = np.array([conc * Vic / 1000, T, E3, BPD_T, BPD_E3, Ternary])
    t = np.array([0, 24])
    results = calc_concentrations(times = t, y0 = y0)

    Ternary_formation_arr[count, 0] = conc
    Ternary_formation_arr[count, 1] = results.y[-1][-1]

Ternary_formation_df = pd.DataFrame(Ternary_formation_arr, columns = ['Conc_BPD_ic', 'Ternary'])
Ternary_formation_df['relative_Ternary'] = Ternary_formation_df['Ternary'] / Ternary_formation_df['Ternary'].max() * 100

plt.rcParams["figure.figsize"] = [7, 5]
plt.rcParams["figure.autolayout"] = True
ax = Ternary_formation_df.plot(x = 'Conc_BPD_ic',
                               xlabel = 'BPD Concentration (nM)',
                               y = 'relative_Ternary',
                               ylabel = '% Relative Ternary Complex',
                               kind = 'line',
                               ylim = (0, 120),
                               logx = True,
                               # title='Ternary complex formation',
                               legend = False)
plt.show()
