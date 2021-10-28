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

n = 3  # set to 0 for ternary formation
MTT_deg = 0.0015
ktransit_UPS = (n + 1) / MTT_deg  # set to 0 for ternary formation
fu_c = np.nan
fu_ec = 1.
fu_ic = 1.
F = np.nan
ka = np.nan
CL = np.nan
Vc = np.nan
Q = np.nan
Vp = np.nan
PS_cell = 1e-12  # set to 0 for ternary formation
PSV_tissue = np.nan
MW_BPD = 947.

# PHYSIOLOGICAL SYSTEM PARAMETERS
kdeg_T = 0.058  # set to 0 for ternary formation
Conc_T_base = 0.001
Conc_E3_base = 0.1
num_cells = 5e3
Vic = 5e-13
Vec = 2e-4
kprod_T = Conc_T_base * Vic * kdeg_T
BW = np.nan

"""INITIAL VALUES"""
BPD_ev = 0
BPD_c = 0
BPD_p = 0
BPD_ec = 0  # nM * Vec / 1000
BPD_ic = 0  # nM * Vic / 1000
T = Conc_T_base * Vic
E3 = Conc_E3_base * Vic
BPD_T = 0
BPD_E3 = 0
Ternary = 0
Ternary_Ubs = [0] * n  # where i = 0 is un-ubiquitinated Ternary

"""KINETIC RATES"""
def dBPD_ecdt(BPD_ec, BPD_ic):
    return -PS_cell * num_cells * ((fu_ec * BPD_ec / Vec) - (fu_ic * BPD_ic / Vic))

def dBPD_icdt(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3):
    return PS_cell * ((fu_ec * BPD_ec / Vec) - (fu_ic * BPD_ic / Vic)) - \
           kon_T_binary * fu_ic * BPD_ic * T / Vic + \
           koff_T_binary * BPD_T - \
           kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \
           koff_E3_binary * BPD_E3 + \
           kdeg_T * BPD_T

def dTargetdt(BPD_ic, T, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    return kprod_T - kdeg_T * T - \
           kon_T_binary * fu_ic * BPD_ic * T / Vic + \
           koff_T_binary * BPD_T - \
           kon_T_ternary * BPD_E3 * T / Vic + \
           koff_T_ternary * (Ternary + np.sum(Ternary_Ubs))

def dE3dt(BPD_ic, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    return -kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \
           koff_E3_binary * BPD_E3 - \
           kon_E3_ternary * BPD_T * E3 / Vic + \
           koff_E3_ternary * (Ternary + np.sum(Ternary_Ubs))

def dBPD_Tdt(BPD_ic, T, E3, BPD_T, Ternary, *Ternary_Ubs):
    return kon_T_binary * fu_ic * BPD_ic * T / Vic - \
           koff_T_binary * BPD_T - \
           kon_E3_ternary * BPD_T * E3 / Vic + \
           koff_E3_ternary * (Ternary + np.sum(Ternary_Ubs)) - \
           kdeg_T * BPD_T

def dBPD_E3dt(BPD_ic, T, E3, BPD_E3, Ternary, *Ternary_Ubs):
    return kon_E3_binary * fu_ic * BPD_ic * E3 / Vic - \
           koff_E3_binary * BPD_E3 - \
           kon_T_ternary * BPD_E3 * T / Vic + \
           (koff_T_ternary + kdeg_T) * (Ternary + np.sum(Ternary_Ubs)) + \
           ktransit_UPS * (Ternary_Ubs[-1] if n > 0 else 0)

def dTernarydt(T, E3, BPD_T, BPD_E3, Ternary):
    return kon_T_ternary * BPD_E3 * T / Vic + \
           kon_E3_ternary * BPD_T * E3 / Vic - \
           (kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS) * Ternary

def dTernary_Ubdt(Ternary_Ub_consec):
    return ktransit_UPS * Ternary_Ub_consec[0] - \
           (kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS) * Ternary_Ub_consec[1]

def rates_ternary_formation(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    BPD_ec_rate = dBPD_ecdt(BPD_ec, BPD_ic)
    BPD_ic_rate = dBPD_icdt(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3)
    T_rate = dTargetdt(BPD_ic, T, BPD_T, BPD_E3, Ternary, *Ternary_Ubs)
    E3_rate = dE3dt(BPD_ic, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs)
    BPD_T_rate = dBPD_Tdt(BPD_ic, T, E3, BPD_T, Ternary, *Ternary_Ubs)
    BPD_E3_rate = dBPD_E3dt(BPD_ic, T, E3, BPD_E3, Ternary, *Ternary_Ubs)
    Ternary_rate = dTernarydt(T, E3, BPD_T, BPD_E3, Ternary)
    if n > 0:  # if there is at least one ubiquitination step
        Ternary_all = np.insert(np.array(Ternary_Ubs), 0, Ternary)  # prepend Ternary to Ternary_Ubs
        Ternary_pairs = np.lib.stride_tricks.sliding_window_view(Ternary_all, 2)  # create array of consecutive pairs
        Ternary_Ubs_rates = np.apply_along_axis(dTernary_Ubdt, 1, Ternary_pairs).tolist()  # apply dTernary_Ubdt() to each pair
    else:
        Ternary_Ubs_rates = []  # no rates for ubiquitinated Ternary complexes if there are none

    all_rates = np.array(
        [BPD_ec_rate, BPD_ic_rate, T_rate, E3_rate, BPD_T_rate, BPD_E3_rate, Ternary_rate] + Ternary_Ubs_rates
    )
    return all_rates

def jac_rates_ternary_formation(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary, *Ternary_Ubs):
    """
    Jacobian of rates with respect to variables.
    dBPD_ecdtdy, dBPD_icdtdy, ..., dTernarydtdy: list < 7 + n >
    dTernary_Ubdtdy_all: list-of-lists < n, 7 + n >
    """
    dBPD_ecdtdy = (
        [
            -PS_cell * num_cells * fu_ec / Vec,
            PS_cell * num_cells * fu_ic / Vic
        ]
        + [0] * (5 + n)  # does not depend on T, E3, BPD_T, BPD_E3, Ternary, Ternary_Ubs
    )
    dBPD_icdtdy = (
        [
            PS_cell * fu_ec / Vec,
            -PS_cell * fu_ic / Vic - kon_T_binary * fu_ic * T / Vic - kon_E3_binary * fu_ic * E3 / Vic,
            -kon_T_binary * fu_ic * BPD_ic / Vic,
            -kon_E3_binary * fu_ic * BPD_ic / Vic,
            koff_T_binary + kdeg_T,
            koff_E3_binary,
            0
        ]
        + [0] * n  # does not depend on Ternary_Ubs
    )
    dTargetdtdy = (
        [
            0,
            -kon_T_binary * fu_ic * T / Vic,
            -kdeg_T - kon_T_binary * fu_ic * BPD_ic / Vic - kon_T_ternary * BPD_E3 / Vic,
            0,
            koff_T_binary,
            -kon_T_ternary * T / Vic,
            koff_T_ternary
        ]
        + [koff_T_ternary] * n
    )
    dE3dtdy = (
        [
            0,
            -kon_E3_binary * fu_ic * E3 / Vic,
            0,
            -kon_E3_binary * fu_ic * BPD_ic / Vic - kon_E3_ternary * BPD_T / Vic,
            -kon_E3_ternary * E3 / Vic,
            koff_E3_binary,
            koff_E3_ternary
        ]
        + [koff_E3_ternary] * n
    )
    dBPD_Tdtdy = (
        [
            0,
            kon_T_binary * fu_ic * T / Vic,
            kon_T_binary * fu_ic * BPD_ic / Vic,
            -kon_E3_ternary * BPD_T / Vic,
            -koff_T_binary - kon_E3_ternary * E3 / Vic - kdeg_T,
            0,
            koff_E3_ternary
        ]
        + [koff_E3_ternary] * n
    )
    dBPD_E3dtdy = (
        [
            0,
            kon_E3_binary * fu_ic * E3 / Vic,
            -kon_T_ternary * BPD_E3 / Vic,
            kon_E3_binary * fu_ic * BPD_ic / Vic,
            0,
            -koff_E3_binary - kon_T_ternary * T / Vic,
            koff_T_ternary + kdeg_T
        ]
        + [koff_T_ternary + kdeg_T] * (n - 1)  # w.r.t. Ternary_Ub_1, ..., Ternary_Ub_<n-1>. If n == 0, then becomes empty list.
        + [koff_T_ternary + kdeg_T + ktransit_UPS] * (1 if n > 0 else 0)  # w.r.t. Ternary_Ub_n
    )
    dTernarydtdy = (
        [
            0,
            0,
            kon_T_ternary * BPD_E3 / Vic,
            kon_E3_ternary * BPD_T / Vic,
            kon_E3_ternary * E3 / Vic,
            kon_T_ternary * T / Vic,
            -(kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS)
        ]
        + [0] * n  # does not depend on Ternary_Ubs
    )
    dTernary_Ubdtdy_all = []  # initialize empty list for dTernary_Ubdt / dy
    if n > 0:  # if there are ubiquitinated Ternary complexes
        for i in range(n):  # for each Ternary complex transit compartment i
            dTernary_Ub_idtdy = [0] * (7 + n)  # initalize zeros list for dTernary_Ub_i / dt / dy
            dTernary_Ub_idtdy[6 + i] = ktransit_UPS  # dTernary_Ub_i / dt / d[Ternary_Ub_<i-1>]
            dTernary_Ub_idtdy[7 + i] = -(kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS)  # dTernary_Ub_i / dt / d[Ternary_Ub_i]
            dTernary_Ubdtdy_all.append(dTernary_Ub_idtdy)  # append dTernary_Ub_i / dt / dy list to list-of-lists

    all_jacs = np.array(  # (7 + n) x (7 + n) array
        [dBPD_ecdtdy, dBPD_icdtdy, dTargetdtdy, dE3dtdy, dBPD_Tdtdy, dBPD_E3dtdy, dTernarydtdy]  # 7 x (7 + n) list-of-lists
        + dTernary_Ubdtdy_all  # n x (7 + n) list-of-lists
    )
    return all_jacs

def calc_concentrations(times, y0):
    def rates(t, y):
        return rates_ternary_formation(*y)

    def jac_rates(t, y):
        return jac_rates_ternary_formation(*y)

    tmin = np.min(times)
    tmax = np.max(times)
    dtimes = times[1:] - times[:-1]  # intervals between times
    max_step = 0.001

    results = integrate.solve_ivp(rates, (tmin, tmax), y0,
                                  method = 'BDF',
                                  max_step = max_step,
                                  t_eval = times,
                                  jac = jac_rates
                                  )

    return results

def plot_concentrations(times, ytimes, show_plot = True):
    results_df = pd.DataFrame(ytimes.T,
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
    results_df['t'] = times

    if show_plot:
        # plt.rcParams["figure.autolayout"] = True
        ax = results_df.plot(x='t',
                             xlabel = 'Time (hours)',
                             ylabel = 'Amount (uM)',
                             kind='bar',
                             stacked=True,
                             logy = False,
                             title='Amounts of species in ternary complex kinetic model',
                             figsize = (12, 8)
                             )
        plt.legend(bbox_to_anchor=(1.0, 1.0))
        plt.show()

    return results_df

"""SIMULATIONS"""
# species amounts at time = 0
y0 = np.array([100 * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
y0
# time steps
t = np.arange(start = 0, stop = 48 + 1, step = 6)
# t = np.array([0, 24, 48])
t

results = calc_concentrations(times = t, y0 = y0)
results

results.message
results.success
np.all(results.y >= 0)

results_df = plot_concentrations(t, results.y)

BPD_total = results_df.filter(regex = '(BPD.*)|(Ternary.*)', axis = 1).sum(axis = 1)
np.allclose(BPD_total, BPD_total[0])

T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)
np.allclose(T_total, T_total[0])

E3_total = results_df.filter(regex = '(.*E3)|(Ternary.*)').sum(axis = 1)
np.allclose(E3_total, E3_total[0])

"""
BARTLETT SUPPLEMENTARY FIGURE 1 (b)
"""
Conc_BPD_ec_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)
Target_deg_arr = np.empty((len(Conc_BPD_ec_arr),2))

for count, conc in enumerate(Conc_BPD_ec_arr):
    y0 = np.array([conc * Vec / 1000, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
    t = np.array([0, 24])
    results = calc_concentrations(times = t, y0 = y0)
    results_df = plot_concentrations(t, results.y, show_plot = False)
    T_total = results_df.filter(regex = '(.*T)|(Ternary.*)').sum(axis = 1)

    Target_deg_arr[count, 0] = conc
    Target_deg_arr[count, 1] = T_total.values[1] / T * 100

Target_deg_df = pd.DataFrame(Target_deg_arr, columns = ['Conc_BPD_ec', 'Target_deg'])

plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.autolayout"] = True
ax = Target_deg_df.plot(
    x = 'Conc_BPD_ec',
    xlabel = 'BPD Concentration (nM)',
    y = 'Target_deg',
    ylabel = '% Baseline Target Protein',
    kind = 'line',
    xlim = (1e-1, 1e5),
    ylim = (0, 120),
    logx = True,
    # title='Ternary complex formation',
    legend = False
)
plt.show()


"""
BARTLETT SUPPLEMENTARY FIGURE 1 (a)
Ternary complex formation only.
"""
Conc_BPD_ic_arr = np.logspace(base = 10.0, start = -1, stop = 5, num = 50)
Ternary_formation_arr = np.empty((len(Conc_BPD_ic_arr),2))

for count, conc in enumerate(Conc_BPD_ic_arr):
    y0 = np.array([BPD_ec, conc * Vic / 1000, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)
    t = np.array([0, 24])
    results = calc_concentrations(times = t, y0 = y0)

    Ternary_formation_arr[count, 0] = conc
    Ternary_formation_arr[count, 1] = results.y[-1][-1]

Ternary_formation_df = pd.DataFrame(Ternary_formation_arr, columns = ['Conc_BPD_ic', 'Ternary'])
Ternary_formation_df['relative_Ternary'] = Ternary_formation_df['Ternary'] / Ternary_formation_df['Ternary'].max() * 100

plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.autolayout"] = True
ax = Ternary_formation_df.plot(
    x = 'Conc_BPD_ic',
    xlabel = 'BPD Concentration (nM)',
    y = 'relative_Ternary',
    ylabel = '% Relative Ternary Complex',
    kind = 'line',
    xlim = (1e-1, 1e5),
    ylim = (0, 120),
    logx = True,
    # title='Ternary complex formation',
    legend = False
)
plt.show()
