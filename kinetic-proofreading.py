import scipy.integrate as integrate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""GLOBAL VARIABLES"""
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

n = 3
MTT_deg = 0.0015
ktransit_UPS = (n + 1) / MTT_deg
fu_c = np.nan
fu_ec = 1.
fu_ic = 1.
F = np.nan
ka = np.nan
CL = np.nan
Vc = np.nan
Q = np.nan
Vp = np.nan
PS_cell = 1e-12
PSV_tissue = np.nan
MW_BPD = 947.

kdeg_T = 0.058
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
BPD_ec = 100 * Vec / 1000  # nM * Vec / 1000
BPD_ic = 0
T = Conc_T_base * Vic
E3 = Conc_E3_base * Vic
BPD_T = 0
BPD_E3 = 0
Ternary = 0
Ternary_Ub_i = np.zeros(n)  # where i = 0 is un-ubiquitinated Ternary

"""KINETIC RATES"""
def dBPD_ecdt(BPD_ec, BPD_ic):
    return -PS_cell * num_cells * ((fu_ec * BPD_ec / Vec) - (fu_ic * BPD_ic / Vic))

def dBPD_icdt(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3):
    return PS_cell * ((fu_ec * BPD_ec / Vec) - (fu_ic * BPD_ic / Vic)) - \
           kon_T_binary * fu_ic * BPD_ic * T / Vic + \
           koff_T_binary * BPD_T - \
           kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \
           koff_E3_binary * BPD_E3 # + \
           # kdeg_T * BPD_T

def dTargetdt(BPD_ic, T, BPD_T, BPD_E3, Ternary):
    return -kon_T_binary * fu_ic * BPD_ic * T / Vic + \
           koff_T_binary * BPD_T - \
           kon_T_ternary * BPD_E3 * T / Vic + \
           koff_T_ternary * Ternary
           # kprod_T - \
           # kdeg_T * T - \
           # koff_T_ternary * np.sum(Ternary_Ub_i)

def dE3dt(BPD_ic, E3, BPD_T, BPD_E3, Ternary):
    return -kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \
            koff_E3_binary * BPD_E3 - \
            kon_E3_ternary * BPD_T * E3 / Vic + \
            koff_E3_ternary * Ternary
            # koff_E3_ternary * np.sum(Ternary_Ub_i)

def dBPD_Tdt(BPD_ic, T, E3, BPD_T, Ternary):
    return kon_T_binary * fu_ic * BPD_ic * T / Vic - \
           koff_T_binary * BPD_T - \
           kon_E3_ternary * BPD_T * E3 / Vic + \
           koff_E3_ternary * Ternary
           # koff_E3_ternary * np.sum(Ternary_Ub_i) - \
           # kdeg_T * BPD_T

def dBPD_E3dt(BPD_ic, T, E3, BPD_E3, Ternary):
    return kon_E3_binary * fu_ic * BPD_ic * E3 / Vic - \
           koff_E3_binary * BPD_E3 - \
           kon_T_ternary * BPD_E3 * T / Vic + \
           koff_T_ternary * Ternary
           # koff_T_ternary * np.sum(Ternary_Ub_i) + \
           # kdeg_T * np.sum(Ternary_Ub_i) + \
           # ktransit_UPS * Ternary_Ub_i[-1]

def dTernarydt(T, E3, BPD_T, BPD_E3, Ternary):
    return kon_T_ternary * BPD_E3 * T / Vic + \
           kon_E3_ternary * BPD_T * E3 / Vic - \
           (koff_T_ternary + koff_E3_ternary) * Ternary
           # (kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS) * Ternary

def dTernary_Ub_idt(Ternary_Ub_i_prev, Ternary_Ub_i):
    return ktransit_UPS * Ternary_Ub_i_prev - \
           (kdeg_T + koff_T_ternary + koff_E3_ternary + ktransit_UPS) * Ternary_Ub_i

def rates_ternary_formation(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary):
    BPD_ec_rate = dBPD_ecdt(BPD_ec, BPD_ic)
    BPD_ic_rate = dBPD_icdt(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3)
    T_rate = dTargetdt(BPD_ic, T, BPD_T, BPD_E3, Ternary)
    E3_rate = dE3dt(BPD_ic, E3, BPD_T, BPD_E3, Ternary)
    BPD_T_rate = dBPD_Tdt(BPD_ic, T, E3, BPD_T, Ternary)
    BPD_E3_rate = dBPD_E3dt(BPD_ic, T, E3, BPD_E3, Ternary)
    Ternary_rate = dTernarydt(T, E3, BPD_T, BPD_E3, Ternary)
    return np.array([BPD_ec_rate, BPD_ic_rate, T_rate, E3_rate, BPD_T_rate, BPD_E3_rate, Ternary_rate])

def jac_rates_ternary_formation(BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary):
    # dBPD_ecdt / dy
    dBPD_ecdtdy = np.array([-PS_cell * num_cells * fu_ec / Vec,
                            PS_cell * num_cells * fu_ic / Vic,
                            0, 0, 0, 0, 0])
    # dBPD_icdt / dy
    dBPD_icdtdy = np.array([PS_cell * fu_ec / Vec,
                            -PS_cell * fu_ic / Vic - kon_T_binary * fu_ic * T / Vic - kon_E3_binary * fu_ic * E3 / Vic,
                            -kon_T_binary * fu_ic * BPD_ic / Vic, -kon_E3_binary * fu_ic * BPD_ic / Vic,
                            koff_T_binary, koff_E3_binary, 0])
    #dTargetdt / dy
    dTargetdtdy = np.array([0, -kon_T_binary * fu_ic * T / Vic,
                            -kon_T_binary * fu_ic * BPD_ic / Vic - kon_T_ternary * BPD_E3 / Vic, 0,
                            koff_T_binary, -kon_T_ternary * T / Vic, koff_T_ternary])
    #dE3dt / dy
    dE3dtdy = np.array([0, -kon_E3_binary * fu_ic * E3 / Vic,
                        0, -kon_E3_binary * fu_ic * BPD_ic / Vic - kon_E3_ternary * BPD_T / Vic,
                        -kon_E3_ternary * E3 / Vic, koff_E3_binary, koff_E3_ternary])
    #dBPD_Tdt / dy
    dBPD_Tdtdy = np.array([0, kon_T_binary * fu_ic * T / Vic,
                           kon_T_binary * fu_ic * BPD_ic / Vic, -kon_E3_ternary * BPD_T / Vic,
                           -koff_T_binary - kon_E3_ternary * E3 / Vic, 0, koff_E3_ternary])
    #dBPD_E3dt / dy
    dBPD_E3dtdy = np.array([0, kon_E3_binary * fu_ic * E3 / Vic,
                            -kon_T_ternary * BPD_E3 / Vic, kon_E3_binary * fu_ic * BPD_ic / Vic,
                            0, -koff_E3_binary - kon_T_ternary * T / Vic, koff_T_ternary])
    # dTernarydt / dy
    dTernarydtdy = np.array([0, 0, kon_T_ternary * BPD_E3 / Vic, kon_E3_ternary * BPD_T / Vic,
                             kon_E3_ternary * E3 / Vic, kon_T_ternary * T / Vic, -(koff_T_ternary + koff_E3_ternary)])

    return np.array([dBPD_ecdtdy, dBPD_icdtdy, dTargetdtdy, dE3dtdy, dBPD_Tdtdy, dBPD_E3dtdy, dTernarydtdy])

def calc_concentrations(times, y0):
    def rates(t, y):
        return rates_ternary_formation(*y)

    def jac_rates(t, y):
        return jac_rates_ternary_formation(*y)

    tmin = np.min(times)
    tmax = np.max(times)
    results = integrate.solve_ivp(rates, (tmin, tmax), y0,
                                  method = 'LSODA',
                                  t_eval = times,
                                  jac = jac_rates
                                  )

    return results

def plot_concentrations(times, ytimes):
    results_df = pd.DataFrame(ytimes.T, columns = ['BPD_ec', 'BPD_ic', 'T', 'E3', 'BPD_T', 'BPD_E3', 'Ternary'])
    results_df['t'] = times

    plt.rcParams["figure.figsize"] = [10, 5]
    plt.rcParams["figure.autolayout"] = True
    ax = results_df.plot(x='t', y=['BPD_T', 'BPD_E3', 'Ternary'], kind='bar', stacked=True, logy = False,
                         title='Amounts of species in ternary complex kinetic model')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Amount (uM)')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()
    return results_df

# species amounts at time = 0
y0 = np.array([BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary])
# time steps
t = np.arange(start = 0, stop = 168 + 1, step = 24)

y0
t

results = calc_concentrations(times = t, y0 = y0)
results.success
results.y

plot_concentrations(t, results.y)
