"""
This module contains functions used to implement a kinetic proofreading model of target protein degradation.
"""
from typing import Optional, Iterable

import numpy as np
from numpy.typing import NDArray, ArrayLike
import pandas as pd
import sklearn
from scipy.integrate import solve_ivp
from scipy.optimize import root
from matplotlib import pyplot as plt

"""
Abbreviations signify as follows:

T: Target
BPD: Bi-functional Protein Degrader
E3: E3 ligase
BPD_T: BPD.T binary complex
BPD_E3: BPD.E3 binary complex
Ternary: T.BPD.E3 ternary complex
T_Ub_i: Target with i Ubiquitin molecules attached
BPD_T_Ub_i: BPD.T binary complex with i Ubiquitin molecules attached
Ternary_Ub_i: T.BPD.E3 ternary complex with i Ubiquitin molecules attached

Functions in KINETIC RATES section must be provided a `params` dictionary with the following fields:

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
kub: ubiquitination rate
kde_ub: de-ubiquitination rate
kdeg_UPS: proteasomal target protein degradation rate
fu_ec: fraction unbound extracellular BPD
fu_ic: fraction unbound intracellular BPD
PS_cell: permeability-surface area product
kprod_T: intrinsic target protein production rate
kdeg_T: intrinsic target protein degradation rate
Conc_T_base: baseline target protein concentration
Conc_E3_base: baseline E3 concentration
num_cells: number of cells in system
Vic: intracellular volume
Vec: extracellular volume
"""

"""
KINETIC RATES
"""


def dBPD_ecdt(BPD_ec: float,
              BPD_ic: float,
              params: dict[str, float]) -> float:
    """Calculates dBPD_ec / dt.

    Parameters
    ----------
    BPD_ec : float
        amount of extracellular BPD

    BPD_ic : float
        amount of intracellular BPD

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dBPD_ec / dt
    """
    return (
            - params['PS_cell'] * params['num_cells'] * (
                (params['fu_ec'] * BPD_ec / params['Vec']) - (params['fu_ic'] * BPD_ic / params['Vic']))
    )


def dBPD_icdt(BPD_ec: float,
              BPD_ic: float,
              T: float,
              T_Ubs: NDArray[np.float64],
              E3: float,
              BPD_T: float,
              BPD_T_Ubs: NDArray[np.float64],
              BPD_E3: float,
              params: dict[str, float]) -> float:
    """Calculates dBPD_ic / dt.

    Parameters
    ----------
    BPD_ec : float
        amount of extracellular BPD

    BPD_ic : float
        amount of intracellular BPD

    T : float
        amount of Target

    T_Ubs : NDArray[np.float64]
        amounts of ubiquitinated Targets

    E3 : float
        amount of E3

    BPD_T : float
        amount of BPD_T

    BPD_T_Ubs : NDArray[np.float64]
        amounts of ubiquitinated BPD_T's

    BPD_E3 : float
        amount of BPD_E3

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dBPD_ic / dt
    """
    return (
            params['PS_cell'] * ((params['fu_ec'] * BPD_ec / params['Vec']) - (params['fu_ic'] * BPD_ic / params['Vic']))
            - params['kon_T_binary'] * params['fu_ic'] * BPD_ic * (T + np.sum(T_Ubs)) / params['Vic']
            + (params['koff_T_binary'] + params['kdeg_T']) * (BPD_T + np.sum(BPD_T_Ubs))
            - params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic']
            + params['koff_E3_binary'] * BPD_E3
            + params['kdeg_UPS'] * (BPD_T_Ubs[-1] if len(BPD_T_Ubs) else 0)
    )


def dTargetdt(BPD_ic: float,
              T: float,
              T_Ubs: NDArray[np.float64],
              BPD_T: float,
              BPD_E3: float,
              Ternary: float,
              params: dict[str, float]) -> float:
    """Calculates dTarget / dt.

    Parameters
    ----------
    BPD_ic : float
        amount of intracellular BPD

    T : float
        amount of Target

    T_Ubs : NDArray[np.float64]
        amounts of ubiquitinated Targets

    BPD_T : float
        amount of BPD_T

    BPD_E3 : float
        amount of BPD_E3

    Ternary : float
        amount of Ternary

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dTarget / dt
    """
    return (
            params['kprod_T'] - params['kdeg_T'] * T
            - params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic']
            + params['koff_T_binary'] * BPD_T
            - params['kon_T_ternary'] * BPD_E3 * T / params['Vic']
            + params['koff_T_ternary'] * Ternary
            + params['kde_ub'] * (T_Ubs[0] if len(T_Ubs) else 0)
    )


def dE3dt(BPD_ic: float,
          E3: float,
          BPD_T: float,
          BPD_T_Ubs: NDArray[np.float64],
          BPD_E3: float,
          Ternary: float,
          Ternary_Ubs: NDArray[np.float64],
          params: dict[str, float]) -> float:
    """Calculates dE3 / dt.

    Parameters
    ----------
    BPD_ic : float
        amount of intracellular BPD

    E3 : float
        amount of E3

    BPD_T : float
        amount of BPD_T

    BPD_T_Ubs : NDArray[np.float64]
        amounts of ubiquitinated BPD_T's

    BPD_E3 : float
        amount of BPD_E3

    Ternary : float
        amount of Ternary

    Ternary_Ubs : NDArray[np.float64]
        amounts of ubiquitinated Ternaries

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dE3 / dt
    """
    return (
            - params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic']
            + params['koff_E3_binary'] * BPD_E3
            - params['kon_E3_ternary'] * E3 * (BPD_T + np.sum(BPD_T_Ubs)) / params['Vic']
            + params['koff_E3_ternary'] * (Ternary + np.sum(Ternary_Ubs))
    )


def dBPD_Tdt(BPD_ic: float,
             T: float,
             E3: float,
             BPD_T: float,
             BPD_T_Ubs: NDArray[np.float64],
             Ternary: float,
             params: dict[str, float]) -> float:
    """Calculates dBPD.T / dt.

    Parameters
    ----------
    BPD_ic : float
        amount of intracellular BPD

    T : float
        amount of Target

    E3 : float
        amount of E3

    BPD_T : float
        amount of BPD_T

    BPD_T_Ubs : NDArray[np.float64]
        amounts of ubiquitinated BPD_T's

    Ternary : float
        amount of Ternary

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dBPD.T / dt
    """
    return (
            params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T / params['Vic']
            - (params['koff_T_binary'] + params['kdeg_T']) * BPD_T
            - params['kon_E3_ternary'] * BPD_T * E3 / params['Vic']
            + params['koff_E3_ternary'] * Ternary
            + params['kde_ub'] * (BPD_T_Ubs[0] if len(BPD_T_Ubs) else 0)
    )


def dBPD_E3dt(BPD_ic: float,
              T: float,
              T_Ubs: NDArray[np.float64],
              E3: float,
              BPD_E3: float,
              Ternary: float,
              Ternary_Ubs: NDArray[np.float64],
              params: dict[str, float]) -> float:
    """Calculates dBPD.E3 / dt.

    BPD_ic : float
        amount of intracellular BPD

    T : float
        amount of Target

    T_Ubs : NDArray[np.float64]
        amounts of ubiquitinated Targets

    E3 : float
        amount of E3

    BPD_E3 : float
        amount of BPD_E3

    Ternary : float
        amount of Ternary

    Ternary_Ubs : NDArray[np.float64]
        amounts of ubiquitinated Ternaries

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dBPD.E3 / dt
    """
    return (
            params['kon_E3_binary'] * params['fu_ic'] * BPD_ic * E3 / params['Vic']
            - params['koff_E3_binary'] * BPD_E3
            - params['kon_T_ternary'] * BPD_E3 * (T + np.sum(T_Ubs)) / params['Vic']
            + (params['koff_T_ternary'] + params['kdeg_T']) * (Ternary + np.sum(Ternary_Ubs))
            + params['kdeg_Ternary'] * (Ternary_Ubs[-1] if len(Ternary_Ubs) else 0)
    )


def dTernarydt(T: float,
               E3: float,
               BPD_T: float,
               BPD_E3: float,
               Ternary: float,
               params: dict[str, float]) -> float:
    """Calculates dTernary / dt.

    BPD_ic : float
        amount of intracellular BPD

    T : float
        amount of Target

    E3 : float
        amount of E3

    BPD_T : float
        amount of BPD_T

    BPD_E3 : float
        amount of BPD_E3

    Ternary : float
        amount of Ternary

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dTernary / dt
    """
    return (
            params['kon_T_ternary'] * BPD_E3 * T / params['Vic']
            + params['kon_E3_ternary'] * BPD_T * E3 / params['Vic']
            - (params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['kub']) * Ternary
    )


def dT_Ubdt(T_Ub_consec_pair: NDArray[np.float64],
            BPD_ic: float,
            BPD_T_Ub_i: float,
            BPD_E3: float,
            Ternary_Ub_i: float,
            i: int,
            params: dict[str, float]) -> float:
    """Calculates dT.Ub.i / dt for i = 1, ..., n.

    Parameters
    ----------
    T_Ub_consec_pair : NDArray[np.float64]
        amounts of T.Ub.i, T.Ub.<i+1> or T.Ub.n, 0

    BPD_ic : float
        amount of intracellular BPD

    BPD_T_Ub_i : float
        amounts of BPD.T.Ub.i

    BPD_E3 : float
        amount of BPD_E3

    Ternary_Ub_i : float
        amount of Ternary.Ub.i

    i : int
        index of T.Ub.i

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dT.Ub.i / dt
    """
    return (
            - (params['kde_ub'] + params['kdeg_T']) * T_Ub_consec_pair[0]
            + params['kde_ub'] * T_Ub_consec_pair[1]
            - params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T_Ub_consec_pair[0] / params['Vic']
            + params['koff_T_binary'] * BPD_T_Ub_i
            - params['kon_T_ternary'] * BPD_E3 * T_Ub_consec_pair[0] / params['Vic']
            + params['koff_T_ternary'] * Ternary_Ub_i
            - params['kdeg_UPS'] * (T_Ub_consec_pair[0] if i == params['n'] else 0)
    )


def dBPD_T_Ubdt(BPD_T_Ub_consec_pair: NDArray[np.float64],
                BPD_ic: float,
                T_Ub_i: float,
                E3: float,
                Ternary_Ub_i: float,
                i: int,
                params: dict[str, float]) -> float:
    """Calculates dBPD.T.Ub.i / dt for i = 1, ..., n.

    Parameters
    ----------
    BPD_T_Ub_consec_pair : NDArray[np.float64]
        amounts of BPD.T.Ub.i, BPD.T.Ub.<i+1> or BPD.T.Ub.n, 0

    BPD_ic : float
        amount of intracellular BPD

    T_Ub_i : float
        amounts of T.Ub.i

    Ternary_Ub_i : float
        amount of Ternary.Ub.i

    i : int
        index of BPD.T.Ub.i

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dBPD.T.Ub.i / dt
    """
    return (
            - (params['kde_ub'] + params['kdeg_T']) * BPD_T_Ub_consec_pair[0]
            + params['kde_ub'] * BPD_T_Ub_consec_pair[1]
            + params['kon_T_binary'] * params['fu_ic'] * BPD_ic * T_Ub_i / params['Vic']
            - params['koff_T_binary'] * BPD_T_Ub_consec_pair[0]
            - params['kon_E3_ternary'] * BPD_T_Ub_consec_pair[0] * E3 / params['Vic']
            + params['koff_E3_ternary'] * Ternary_Ub_i
            - params['kdeg_UPS'] * (BPD_T_Ub_consec_pair[0] if i == params['n'] else 0)
    )


def dTernary_Ubdt(Ternary_Ub_consec_pair: NDArray[np.float64],
                  T_Ub_i: float,
                  E3: float,
                  BPD_T_Ub_i: float,
                  BPD_E3: float,
                  i: int,
                  params: dict[str, float]) -> float:
    """Calculates dTernary_Ub_i / dt for i = 1, ... n.

    Parameters
    ----------
    Ternary_Ub_consec_pair : NDArray[np.float64]
        amounts of Ternary.Ub.<i-1>, Ternary.Ub.i or Ternary, Ternary.Ub.1

    T_Ub_i : float
        amounts of T.Ub.i

    E3 : float
        amount of E3

    BPD_T_Ub_i : float
        amount of BPD.T.Ub.i

    BPD_E3 : float
        amount of BPD_E3

    i : int
        index of Ternary.Ub.i

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    float
        dTernary.Ub.i / dt
    """
    return (
            params['kub'] * Ternary_Ub_consec_pair[0]
            + params['kon_T_ternary'] * BPD_E3 * T_Ub_i / params['Vic']
            + params['kon_E3_ternary'] * BPD_T_Ub_i * E3 / params['Vic']
            - (params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary']) * Ternary_Ub_consec_pair[1]
            - (params['kdeg_Ternary'] if i == params['n'] else params['kub']) * Ternary_Ub_consec_pair[1]
    )


def kinetic_rates(y: NDArray[np.float64], params: dict[str, float]) -> NDArray[np.float64]:
    """Calculates rates of change for species in PROTAC-induced target protein degradation.

    Parameters
    ----------
    y : NDArray[np.float64]
        amounts of species in this order:
            BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary,
            T_Ub_1, ..., T_Ub_n, BPD_T_Ub_1, ..., BPD_T_Ub_n, Ternary_Ub_1, ..., Ternary_Ub_n

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    NDArray[np.float64]
        rates of change with respect to time
    """
    # unpack the first 7 species
    BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
    # split the following species into three equally-sized arrays
    Ub_species = np.array_split(y[7:], 3)
    T_Ubs, BPD_T_Ubs, Ternary_Ubs = Ub_species[0], Ub_species[1], Ub_species[2]

    BPD_ec_rate = dBPD_ecdt(BPD_ec, BPD_ic, params)
    BPD_ic_rate = dBPD_icdt(BPD_ec, BPD_ic, T, T_Ubs, E3, BPD_T, BPD_T_Ubs, BPD_E3, params)
    T_rate = dTargetdt(BPD_ic, T, T_Ubs, BPD_T, BPD_E3, Ternary, params)
    E3_rate = dE3dt(BPD_ic, E3, BPD_T, BPD_T_Ubs, BPD_E3, Ternary, Ternary_Ubs, params)
    BPD_T_rate = dBPD_Tdt(BPD_ic, T, E3, BPD_T, BPD_T_Ubs, Ternary, params)
    BPD_E3_rate = dBPD_E3dt(BPD_ic, T, T_Ubs, E3, BPD_E3, Ternary, Ternary_Ubs, params)
    Ternary_rate = dTernarydt(T, E3, BPD_T, BPD_E3, Ternary, params)
    T_Ubs_rates: list[float] = []
    BPD_T_Ubs_rates: list[float] = []
    Ternary_Ubs_rates: list[float] = []

    if params['n'] > 0:
        # there is at least one ubiquitination step
        T_all = np.append(T_Ubs, values=0)
        T_pairs = np.lib.stride_tricks.sliding_window_view(T_all, window_shape=2)

        BPD_T_all = np.append(BPD_T_Ubs, values=0)
        BPD_T_pairs = np.lib.stride_tricks.sliding_window_view(BPD_T_all, window_shape=2)

        Ternary_all = np.insert(Ternary_Ubs, obj=0, values=Ternary)  # prepend Ternary to Ternary_Ubs
        Ternary_pairs = np.lib.stride_tricks.sliding_window_view(Ternary_all, window_shape=2)

        for i in range(params['n']):
            dT_Ub_idt = dT_Ubdt(T_pairs[i], BPD_ic, BPD_T_Ubs[i], BPD_E3, Ternary_Ubs[i], i + 1, params)
            T_Ubs_rates.append(dT_Ub_idt)

            dBPD_T_Ub_idt = dBPD_T_Ubdt(BPD_T_pairs[i], BPD_ic, T_Ubs[i], E3, Ternary_Ubs[i], i + 1, params)
            BPD_T_Ubs_rates.append(dBPD_T_Ub_idt)

            dTernary_Ub_idt = dTernary_Ubdt(Ternary_pairs[i], T_Ubs[i], E3, BPD_T_Ubs[i], BPD_E3, i + 1, params)
            Ternary_Ubs_rates.append(dTernary_Ub_idt)

    # list concatenation
    all_rates = np.array(
        [BPD_ec_rate, BPD_ic_rate, T_rate, E3_rate, BPD_T_rate, BPD_E3_rate, Ternary_rate]
        + T_Ubs_rates
        + BPD_T_Ubs_rates
        + Ternary_Ubs_rates
    )

    return all_rates


def jac_kinetic_rates(y: NDArray[np.float64], params: dict[str, float]) -> NDArray[np.float64]:
    """Calculates Jacobian of system of rate equations.

    Calculates M x M Jacobian J of M rate equations with respect to M species
    where J[i,j] is d(dyi / dt) / dyj.

    Must have identical call signature as kinetic_rates().

    Parameters
    ----------
    y : NDArray[np.float64]
        amounts of species in this order:
            BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary,
            T_Ub_1, ..., T_Ub_n, BPD_T_Ub_1, ..., BPD_T_Ub_n, Ternary_Ub_1, ..., Ternary_Ub_n

    params : dict[str, float]
        kinetic rate constants and model parameters

    Returns
    -------
    NDArray[np.float64]
        Jacobian matrix of system of rate equations
    """
    # unpack the first 7 species
    BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary = y[0], y[1], y[2], y[3], y[4], y[5], y[6]
    # split the following species into three equally-sized arrays
    Ub_species = np.array_split(y[7:], 3)
    T_Ubs, BPD_T_Ubs, Ternary_Ubs = Ub_species[0], Ub_species[1], Ub_species[2]

    n_Ub_species = len(y[7:]) # total number of ubiquitinated species

    # n_T_Ubs = n_BPD_T_Ubs = n_Ternary_Ubs
    # the different variable names may help explain indices of Jacobian values
    n_T_Ubs = len(T_Ubs)
    n_BPD_T_Ubs = len(BPD_T_Ubs)
    n_Ternary_Ubs = len(Ternary_Ubs)

    # dBPD_ec/dt doesn't depend on T, E3, BPD_T, BPD_E3, Ternary, T_Ubs, BPD_T_Ubs, Ternary_Ubs
    dBPD_ecdtdy = (
            [
                -params['PS_cell'] * params['num_cells'] * params['fu_ec'] / params['Vec'],
                params['PS_cell'] * params['num_cells'] * params['fu_ic'] / params['Vic']
            ]
            + [0] * (5 + n_Ub_species)
    )

    dBPD_icdtdy = (
            [
                params['PS_cell'] * params['fu_ec'] / params['Vec'],
                -params['PS_cell'] * params['fu_ic'] / params['Vic']
                - params['kon_T_binary'] * params['fu_ic'] * (T + np.sum(T_Ubs)) / params['Vic']
                - params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
                -params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
                -params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
                params['koff_T_binary'] + params['kdeg_T'],
                params['koff_E3_binary'],
                0
            ]
            + [-params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic']] * n_T_Ubs
            + [params['koff_T_binary'] + params['kdeg_T']] * (n_BPD_T_Ubs - 1)
            + [params['koff_T_binary'] + params['kdeg_T'] + params['kdeg_UPS']] * (1 if n_BPD_T_Ubs else 0)
            + [0] * n_Ternary_Ubs  # dBPD_ic/dt does not depend on Ternary_Ubs
    )

    dTargetdtdy = (
            [
                0,
                -params['kon_T_binary'] * params['fu_ic'] * T / params['Vic'],
                -params['kdeg_T']
                - params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic']
                - params['kon_T_ternary'] * BPD_E3 / params['Vic'],
                0,
                params['koff_T_binary'],
                -params['kon_T_ternary'] * T / params['Vic'],
                params['koff_T_ternary']
            ]
            + [params['kde_ub']] * (1 if n_T_Ubs else 0)
            + [0] * (n_T_Ubs - 1)
            + [0] * (n_BPD_T_Ubs + n_Ternary_Ubs)
    )

    dE3dtdy = (
            [
                0,
                -params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
                0,
                -params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic']
                - params['kon_E3_ternary'] * (BPD_T + np.sum(BPD_T_Ubs)) / params['Vic'],
                -params['kon_E3_ternary'] * E3 / params['Vic'],
                params['koff_E3_binary'],
                params['koff_E3_ternary']
            ]
            + [0] * n_T_Ubs
            + [-params['kon_E3_ternary'] * E3 / params['Vic']] * n_BPD_T_Ubs
            + [params['koff_E3_ternary']] * n_Ternary_Ubs
    )

    dBPD_Tdtdy = (
            [
                0,
                params['kon_T_binary'] * params['fu_ic'] * T / params['Vic'],
                params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
                -params['kon_E3_ternary'] * BPD_T / params['Vic'],
                -(params['koff_T_binary'] + params['kdeg_T'])
                - params['kon_E3_ternary'] * E3 / params['Vic'],
                0,
                params['koff_E3_ternary']
            ]
            + [0] * n_T_Ubs
            + [params['kde_ub']] * (1 if n_BPD_T_Ubs else 0)
            + [0] * (n_BPD_T_Ubs - 1)
            + [0] * n_Ternary_Ubs
    )

    dBPD_E3dtdy = (
            [
                0,
                params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
                -params['kon_T_ternary'] * BPD_E3 / params['Vic'],
                params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
                0,
                -params['koff_E3_binary']
                - params['kon_T_ternary'] * (T + np.sum(T_Ubs)) / params['Vic'],
                params['koff_T_ternary'] + params['kdeg_T']
            ]
            + [-params['kon_T_ternary'] * BPD_E3 / params['Vic']] * n_T_Ubs
            + [0] * n_BPD_T_Ubs
            + [params['koff_T_ternary'] + params['kdeg_T']] * (n_Ternary_Ubs - 1)
            + [params['koff_T_ternary'] + params['kdeg_T'] + params['kdeg_Ternary']] * (1 if n_Ternary_Ubs else 0)
    )

    dTernarydtdy = (
            [
                0,
                0,
                params['kon_T_ternary'] * BPD_E3 / params['Vic'],
                params['kon_E3_ternary'] * BPD_T / params['Vic'],
                params['kon_E3_ternary'] * E3 / params['Vic'],
                params['kon_T_ternary'] * T / params['Vic'],
                -(params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['kub'])
            ]
            + [0] * n_Ub_species  # dTernary/dt doesn't depend on T_Ubs, BPD_T_Ubs, Ternary_Ubs
    )

    dT_Ubdtdy_all = []  # initialize empty list for (dT_Ub/dt) / dy
    dBPD_T_Ubdtdy_all = []  # initialize empty list for (dBPD_T_Ub/dt) / dy
    dTernary_Ubdtdy_all = []  # initialize empty list for (dTernary_Ub/dt) / dy
    if params['n'] > 0:
        # if there are ubiquitination steps
        for i in range(params['n']):  # for each ubiquitination step
            # initalize list of zeros for (dUb.i/dt) / dy
            dT_Ubdtdy = [0] * (7 + n_Ub_species)
            dBPD_T_Ubdtdy = [0] * (7 + n_Ub_species)
            dTernary_Ub_idtdy = [0] * (7 + n_Ub_species)

            dT_Ubdtdy[1] = -params['kon_T_binary'] * params['fu_ic'] * T_Ubs[i] / params['Vic']  # (dT.Ubi/dt) / dBPD_ic
            dT_Ubdtdy[5] = -params['kon_T_ternary'] * T_Ubs[i] / params['Vic']  # (dT.Ubi/dt) / dBPD_E3
            dT_Ubdtdy[7 + i] = (
                    -(params['kde_ub'] + params['kdeg_T'])
                    - params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic']
                    - params['kon_T_ternary'] * BPD_E3 / params['Vic']
                    - (params['kdeg_UPS'] if i == (n_T_Ubs - 1) else 0)
            )  # (dT.Ubi/dt) / dT.Ubi
            if i < (n_T_Ubs - 1):
                dT_Ubdtdy[7 + i + 1] = params['kde_ub']  # (dT.Ubi/dt) / dT.Ub<i+1>

            dT_Ubdtdy[7 + n_T_Ubs + i] = params['koff_T_binary']  # (dT.Ubi/dt) / dBPD.T.Ubi
            dT_Ubdtdy[7 + n_T_Ubs + n_BPD_T_Ubs + i] = params['koff_T_ternary']  # (dT.Ubi/dt) / dTernary.Ubi

            dBPD_T_Ubdtdy[1] = params['kon_T_binary'] * params['fu_ic'] * T_Ubs[i] / params['Vic']  # (dBPD.T.Ubi/dt) / dBPD_ic
            dBPD_T_Ubdtdy[3] = -params['kon_E3_ternary'] * BPD_T_Ubs[i] / params['Vic']  # (dBPD.T.Ubi/dt) / dE3
            dBPD_T_Ubdtdy[7 + i] = params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic']  # (dBPD.T.Ubi/dt) / dT.Ubi
            dBPD_T_Ubdtdy[7 + n_T_Ubs + i] = (
                    -(params['kde_ub'] + params['kdeg_T'] + params['koff_T_binary'])
                    - params['kon_E3_ternary'] * E3 / params['Vic']
                    - (params['kdeg_UPS'] if i == (n_BPD_T_Ubs - 1) else 0)
            )  # (dBPD.T.Ubi/dt) / dBPD.T.Ubi
            if i < (n_BPD_T_Ubs - 1):
                dBPD_T_Ubdtdy[7 + n_T_Ubs + i + 1] = params['kde_ub']  # (dBPD.T.Ubi/dt) / dBPD.T.Ub<i+1>

            dBPD_T_Ubdtdy[7 + n_T_Ubs + n_BPD_T_Ubs + i] = params['koff_E3_ternary']  # (dBPD.T.Ubi/dt) / dTernary.Ubi

            dTernary_Ub_idtdy[3] = params['kon_E3_ternary'] * BPD_T_Ubs[i] / params['Vic']  # (dTernary.Ubi/dt) / dE3
            dTernary_Ub_idtdy[5] = params['kon_T_ternary'] * T_Ubs[i] / params['Vic']  # (dTernary.Ubi/dt) / dBPD.E3
            dTernary_Ub_idtdy[7 + i] = params['kon_T_ternary'] * BPD_E3 / params['Vic']  # (dTernary.Ubi/dt) / dT.Ubi
            dTernary_Ub_idtdy[7 + n_T_Ubs + i] = params['kon_E3_ternary'] * E3 / params['Vic']  # (dTernary.Ubi/dt) / dBPD.T.Ubi
            if i == 0:
                dTernary_Ub_idtdy[6] = params['kub']
            else:
                dTernary_Ub_idtdy[7 + n_T_Ubs + n_BPD_T_Ubs + i - 1] = params['kub']  # (dTernary_Ub_i/dt) / dTernary_Ub_<i-1>

            dTernary_Ub_idtdy[7 + n_T_Ubs + n_BPD_T_Ubs + i] = (
                    -(params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'])
                    - (params['kdeg_Ternary'] if i == (n_Ternary_Ubs - 1) else params['kub'])
            )  # (dTernary_Ub_i/dt) / dTernary_Ub_i

            dT_Ubdtdy_all.append(dT_Ubdtdy)
            dBPD_T_Ubdtdy_all.append(dBPD_T_Ubdtdy)
            dTernary_Ubdtdy_all.append(dTernary_Ub_idtdy)

    # (7 + 3*n) x (7 + 3*n) array
    all_jacs = np.array(
        [dBPD_ecdtdy, dBPD_icdtdy, dTargetdtdy, dE3dtdy, dBPD_Tdtdy, dBPD_E3dtdy,
         dTernarydtdy]  # 7 x (7 + 3*n) list of lists
        + dT_Ubdtdy_all  # n x (7 + 3*n) list of lists
        + dBPD_T_Ubdtdy_all  # n x (7 + 3*n) list of lists
        + dTernary_Ubdtdy_all  # n x (7 + 3*n) list of lists
    )

    return all_jacs


"""
SOLVE SYSTEM OF ODES
"""


def initial_values(params: dict[str, float], BPD_ec: float = 0, BPD_ic: float = 0) -> NDArray[float]:
    """Returns array of initial values for species amounts.

    Parameters
    ----------
    params : dict[str, float]
        model parameters.

    BPD_ec : float
        initial amount of extracellular BPD

    BPD_ic : float
        initial amount of intracellular BPD

    Returns
    -------
    NDArray[float]
        initial values for system species amounts
    """
    T = params['Conc_T_base'] * params['Vic']
    E3 = params['Conc_E3_base'] * params['Vic']
    BPD_T = 0
    BPD_E3 = 0
    Ternary = 0
    T_Ubs = [0] * params['n']
    BPD_T_Ubs = [0] * params['n']
    Ternary_Ubs = [0] * params['n']
    return np.array([BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + T_Ubs + BPD_T_Ubs + Ternary_Ubs)


def calc_concentrations(t_eval: ArrayLike,
                        y0: NDArray[float],
                        params: dict[str, float],
                        max_step: float = np.inf,
                        half_maximal_target: Optional[float] = None) -> sklearn.utils.Bunch:
    """Solves the initial value problem for target protein degradation.

    Computes the amounts of species in the system over time. It is recommended to set max_step to a small value
    (e.g. 0.001) for systems with small initial values.

    Parameters
    ----------
    t_eval : ArrayLike
        time points at which to store computed solution

    y0 : NDArray[float]
        initial values of species amounts in this order
            BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary,
            T_Ub_1, ..., T_Ub_n, BPD_T_Ub_1, ..., BPD_T_Ub_n, Ternary_Ub_1, ..., Ternary_Ub_n

    params : dict[str, float]
        kinetic rate constants and model parameters for rate equations

    max_step : float
        maximum allowed step size for solver.

    half_maximal_target : float
        amount of total Target that is halfway between baseline and steady state amounts

    Returns
    -------
    sklearn.utils.Bunch
        Solution returned by scipy.integrate.solve_ivp().

        The `y` field contains the solution, where rows are species and columns are time points.
    """

    def rates(t, y, params, half_maximal_target):
        """Returns ODEs evaluated at (t, y).

        Must have call signature func(t, y, *args).
        """
        return kinetic_rates(y, params)

    def jac_rates(t, y, params, half_maximal_target):
        """Returns Jacobian of ODEs evaluated at (t, y).

        Must have identical call signature as rates().
        """
        return jac_kinetic_rates(y, params)

    def t_half_event(t, y, params, half_maximal_target) -> float:
        """Returns difference between total amount of Target in system at time t
        and half-maximal degradation amount.

        Must have identical call signature as rates().
        """
        total_target_at_t = np.sum(np.concatenate((y[[2, 4]], y[6:])))  # total amount of Target in system at time t
        return total_target_at_t - half_maximal_target

    t_half_event.terminal = True  # if t_half_event() changes sign, solver will terminate
    t_max = np.max(t_eval)

    result = solve_ivp(
        fun=rates,
        t_span=(0, t_max),
        y0=y0,
        method='BDF',
        t_eval=t_eval,
        events=t_half_event if half_maximal_target is not None else None,
        args=(params, half_maximal_target,),
        max_step=max_step,
        jac=jac_rates
    )

    if not np.all(result.y >= 0):
        print("scipy.integrate.solve_ivp() produces negative solutions. Trying smaller max_step size...")
        max_step = max_step / 2  # halve the max_step
        return calc_concentrations(t_eval=t_eval,
                                   y0=y0,
                                   params=params,
                                   max_step=max_step,
                                   half_maximal_target=half_maximal_target
                                   )

    return result


def solve_steady_state(initial_guess: NDArray[float], params: dict[str, float]) -> NDArray[float]:
    """Solves system steady state.

    Computes the amounts of species at steady state, at which all kinetic rate equations equal 0.

    Parameters
    ----------
    initial_guess : NDArray[float]
        initial guess for steady state solution

    params : dict[str, float]
        kinetic rate constants and model parameters for rate equations

    Returns
    -------
    NDArray[float]
        species amounts at steady state
    """
    methods: list[str] = ['hybr', 'lm']  # options for `method` arg of scipy.optimize.root()

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
            fun=kinetic_rates,
            x0=initial_guess,
            args=(params,),
            method=m,
            jac=jac_kinetic_rates,
            options=options
        )
        if roots.success and np.all(roots.x >= 0):
            break

    if (not roots.success) or np.any(roots.x < 0):
        print("scipy.optimize.root() message: " + roots.message)
        # return None

    return roots.x


def calc_Dmax(y0: NDArray,
              params: dict[str, float],
              initial_guess: Optional[NDArray] = None,
              t_eval: Optional[ArrayLike] = None) -> float:
    """Calculates Dmax.

    Calculates the maximal target protein degradation achieved as a percentage of baseline total Target amount.

    Parameters
    ----------
    y0 : NDArray
        initial values of species amounts

    params : dict[str, float]
        kinetic rate constants and model parameters for rate equations

    initial_guess : Optional[NDArray]
        initial guess for steady state solution

    t_eval : ArrayLike
        argument passed to calc_concentrations() if initial_guess is not provided

    Returns
    -------
    float
        percent maximal target protein degradation relative to baseline total Target amount
    """
    if initial_guess is None:
        print("Solving IVP because no initial guess provided.")
        result = calc_concentrations(t_eval=t_eval, y0=y0, params=params, max_step=0.001)
        initial_guess = result.y[:, -1]  # system state at final time point

    steady_state = solve_steady_state(initial_guess=initial_guess, params=params)

    total_target_steady_state = np.sum(np.concatenate((steady_state[[2, 4]], steady_state[6:])))
    total_target_baseline = np.sum(np.concatenate((y0[[2, 4]], y0[6:])))

    Dmax = (1 - total_target_steady_state / total_target_baseline) * 100
    return Dmax


def calc_t_half(t_eval: ArrayLike, y0: NDArray, params: dict[str, float]) -> float:
    """Computes the time at which the amount of total Target reaches half-maximal degradation amount.

    Parameters
    ----------
    t_eval : ArrayLike
        time points within which to search for solution

    y0 : NDArray
        initial values of species amounts

    params : dict[str, float]
        kinetic rate constants and model parameters for rate equations

    Returns
    -------
    float
        Time at which the amount of total Target reaches half-maximal degradation amount.
    """
    result = calc_concentrations(t_eval=t_eval, y0=y0, params=params, max_step=0.001)
    initial_guess = result.y[:, -1]  # system state at the last time point

    steady_state = solve_steady_state(initial_guess=initial_guess, params=params)

    total_target_steady_state: float = np.ndarray.sum(np.concatenate((steady_state[[2, 4]], steady_state[6:])))
    total_target_baseline: float = np.ndarray.sum(np.concatenate((y0[[2, 4]], y0[6:])))

    half_maximal_target: float = (total_target_baseline + total_target_steady_state) * 0.5

    result_with_events = calc_concentrations(
        t_eval=t_eval,
        y0=y0, params=params,
        max_step=0.001,
        half_maximal_target=half_maximal_target
    )

    t_half = result_with_events.t_events[0][0]  # first event type, first time point
    return t_half


def calc_degradation_curve(t_eval: ArrayLike,
                           params: dict[str, float],
                           initial_BPD_ec_conc: float = 0,
                           initial_BPD_ic_conc: float = 0,
                           return_only_final_state: bool = True) -> pd.DataFrame:
    """Calculates target protein degradation, ternary complex formation, and Dmax.

    Computes the amounts of species in a system over time starting from initial values.

    Parameters
    ----------
    t_eval : ArrayLike
        Time points at which to store computed solution.

    params : dict[str, float]
        Kinetic rate constants and model parameters for rate equations.

    initial_BPD_ec_conc : float
        Initial value of extracellular BPD concentration.

    initial_BPD_ic_conc : float
        Initial value of intracellular BPD concentration.

    return_only_final_state : bool
        Whether to return only final state of system. Default is True.

    Returns
    -------
    pd.DataFrame
        Solutions at time points.

        ======================  ==============================================================================
        t                       time point
        initial_BPD_ec_conc     initial extracellular PROTAC concentration
        initial_BPD_ic_conc     initial intracellular PROTAC concentration
        BPD_ec                  extracellular PROTAC amount
        BPD_ic                  intracellular unbound PROTAC amount
        T                       unbound target amount
        E3                      unbound E3 amount
        BPD_T                   target-PROTAC binary complex amount
        BPD_E3                  E3-PROTAC binary complex amount
        Ternary                 ternary complex amount
        T_Ub_<i>                i-ubiquitinated target amount
        BPD_T_Ub_<i>            i-ubiquitinated target-PROTAC binary complex amount
        Ternary_Ub_<i>          i-ubiquitinated ternary complex amount
        percent_degradation     percent degradation relative to baseline target amount
        relative_target         percent total target relative to baseline target amount
        relative_naked_ternary  percent un-ubiquitinated ternary complex relative to baseline target amount
        relative_all_ternary    percent all ternary complex relative to baseline target amount
        degradation_rate        rate of change in total target
        naked_ternary_rate      rate of change in un-ubiquitinated ternary complex
        poly_ub_target_rate     rate of change in fully ubiquitinated target and target-PROTAC binary complex
        poly_ub_ternary_rate    rate of change in fully ubiquitinated ternary complex
        total_target            total target amount
        total_target_ub         total ubiquitinated target and target-PROTAC binary complex amount
        total_ternary           total ternary complex amount
        total_bpd_ic            total intracellular PROTAC (including complexes) amount
        total_poly_ub_target    total fully ubiquitinated target and target-PROTAC binary complex amount
        total_poly_ub_ternary   total fully ubiquitinated ternary complex amount
        ======================  ==============================================================================
    """

    # initial values
    initial_BPD_ec: float = initial_BPD_ec_conc * params['Vec']
    initial_BPD_ic: float = initial_BPD_ic_conc * params['Vic']
    y0: NDArray[float] = initial_values(params=params, BPD_ec=initial_BPD_ec, BPD_ic=initial_BPD_ic)

    # solve system of ODEs
    concentrations: sklearn.utils.Bunch
    concentrations = calc_concentrations(t_eval=t_eval, y0=y0, params=params, max_step=0.001)
    assert concentrations.success, "Integration by ODE solver failed."

    # format simulation results as DataFrame
    concentrations_df: pd.DataFrame
    concentrations_df = dataframe_concentrations(solve_ivp_result=concentrations, num_Ub_steps=params['n'])

    # run unit tests
    assert passes_unit_tests(concentrations_df), "One or more unit tests failed."

    # calculate target protein degradation and ternary complex formation
    total_target_baseline: float = np.ndarray.sum(np.concatenate((y0[[2, 4]], y0[6:])))

    # # calculate Dmax
    # Dmax: float = calc_Dmax(y0=y0, params=params, initial_guess=concentrations.y[:, -1])
    # average_relative_T = (relative_T.min() + relative_T.max()) / 2  # average of min and max Target degradation seen so far
    # relative_T_index = pd.Index(relative_T)  # index object
    # # let initial guess for steady state be system near half Target degradation
    # initial_guess_idx = relative_T_index.get_loc(average_relative_T, method = 'nearest')
    # x0 = concentrations.y[:, initial_guess_idx]

    # # calculate total intracellular species amounts
    # bpd_totals_over_time: pd.Series = concentrations_df.filter(regex='(^(?!BPD_ec).*BPD.*)|(Ternary.*)').sum(axis=1)
    # t_ub_totals_over_time: pd.Series = concentrations_df.filter(regex='.*T_Ub.*').sum(axis=1)
    # poly_ub_target_totals_over_time: pd.Series = concentrations_df.filter(regex=f".*T_Ub_{params['n']}").sum(axis=1)
    # poly_ub_ternary_totals_over_time: pd.Series = concentrations_df[f"Ternary_Ub_{params['n']}"]

    # calculate degradation rates
    # if r < 0 : net loss in total target
    #    r = 0 : no net change in total target
    #    r > 0 : net gain in total target
    # net change in total target is the sum of rates of all species containing target
    # dT/dt + dBPD.T/dt + dTernary/dt + dT.Ub/dt + dBPD.T.Ub/dt + dTernary.Ub/dt

    # reminder: `y` object from solve_ivp() result is an array where rows: species and cols: time
    rates_at_t = np.apply_along_axis(func1d=kinetic_rates, axis=0, arr=concentrations.y, params=params)
    # select and concatenate the rows for species that involve target
    target_species_rates = np.concatenate((rates_at_t[[2, 4], :], rates_at_t[6:, :]))
    # sum the rates of change for species involving target at each time point
    degradation_rates = pd.Series(np.sum(target_species_rates, axis=0))
    # select the row for rate of change in un-ubiquitinated ternary complex
    naked_ternary_rates = pd.Series(rates_at_t[6, :])
    # select and sum the rows for rates of change in fully ubiquitinated target and target-PROTAC at each time point
    poly_ub_target_rates = pd.Series(np.sum(rates_at_t[[6 + params['n'], 6 + 2 * params['n']], :], axis=0)) if params['n'] > 0 else None
    # select the row for rate of change in fully ubiquitinated ternary complex
    poly_ub_ternary_rates = pd.Series(rates_at_t[-1, :]) if params['n'] > 0 else None
    assert check_target_degradation_rates(
        params=params,
        degradation_from_ode=degradation_rates,
        total_target=concentrations_df.filter(regex='.*T.*').sum(axis=1),
        total_poly_ub_target=concentrations_df.filter(regex=f".*T_Ub_{params['n']}").sum(axis=1),
        total_poly_ub_ternary=concentrations_df[f"Ternary_Ub_{params['n']}"]
    )

    # calculate simulation result metrics
    metrics_df = pd.DataFrame({
        'percent_degradation': 100 - concentrations_df.filter(regex='.*T.*').sum(axis=1) / total_target_baseline * 100,
        'relative_target': concentrations_df.filter(regex='.*T.*').sum(axis=1) / total_target_baseline * 100,
        'relative_naked_ternary': concentrations_df['Ternary'] / total_target_baseline * 100,
        'relative_all_ternary': concentrations_df.filter(regex='Ternary.*').sum(axis=1) / total_target_baseline * 100,
        # 'Dmax': Dmax,
        'degradation_rate': degradation_rates,
        'naked_ternary_rate': naked_ternary_rates,
        'poly_ub_target_rate': poly_ub_target_rates,
        'poly_ub_ternary_rate': poly_ub_ternary_rates,
        'total_target': concentrations_df.filter(regex='.*T.*').sum(axis=1),
        'total_target_ub': concentrations_df.filter(regex='.*T_Ub.*').sum(axis=1),
        'total_ternary': concentrations_df.filter(regex='Ternary.*').sum(axis=1),
        'total_bpd_ic': concentrations_df.filter(regex='(^(?!BPD_ec).*BPD.*)|(Ternary.*)').sum(axis=1),
        'total_poly_ub_target': concentrations_df.filter(regex=f".*T_Ub_{params['n']}").sum(axis=1),
        'total_poly_ub_ternary': concentrations_df[f"Ternary_Ub_{params['n']}"]
    })

    # target_totals_over_time: pd.Series = concentrations_df.filter(regex='.*T.*').sum(axis=1)
    # ternary_totals_over_time: pd.Series = concentrations_df['Ternary']  # amounts of naked Ternary
    # all_ternary_totals_over_time: pd.Series = concentrations_df.filter(regex='Ternary.*').sum(axis=1)
    #
    # relative_target: pd.Series = target_totals_over_time / total_target_baseline * 100  # percent total Target relative to baseline total Target
    # relative_ternary: pd.Series = ternary_totals_over_time / total_target_baseline * 100  # percent naked Ternary relative to baseline total Target
    # relative_all_ternary: pd.Series = all_ternary_totals_over_time / total_target_baseline * 100  # percent total Ternary relative to baseline total Target
    #
    # degradation: pd.Series = 100 - relative_target

    # create simulation metadata DataFrame
    metadata_df = pd.DataFrame({
        't': pd.Series(t_eval),
        'initial_BPD_ec_conc': initial_BPD_ec_conc,
        'initial_BPD_ic_conc': initial_BPD_ic_conc
    })

    # create result DataFrame
    result = pd.concat([metadata_df, concentrations_df, metrics_df], axis=1)

    if return_only_final_state:
        return result.iloc[-1:]  # return system state only at final time point

    return result


def check_target_degradation_rates(params: dict[str, float],
                                   degradation_from_ode: Iterable[float],
                                   total_target: Iterable[float],
                                   total_poly_ub_target: Iterable[float],
                                   total_poly_ub_ternary: Iterable[float]) -> bool:
    """Checks whether simulated instantaneous degradation rates equal instantaneous degradation rates from ODEs.

    Parameters
    ----------
    params : dict[str, float]
        Kinetic rate constants and model parameters.
    degradation_from_ode : Iterable[float]
        Instantaneous degradation rates calculated from ODEs.
    total_target : Iterable[float]
        Instantaneous total target amounts.
    total_poly_ub_target : Iterable[float]
        Instantaneous total poly-ubiquitinated target amounts.
    total_poly_ub_ternary : Iterable[float]
        Instantaneous total poly-ubiquitinated ternary amounts.

    Returns
    -------
    bool
        True if simulated instantaneous degradation rates equal ODE degradation rates.
    """
    degradation_from_sim = (
        params['kprod_T']
        - params['kdeg_T'] * total_target
        - params['kdeg_UPS'] * total_poly_ub_target
        - params['kdeg_Ternary'] * total_poly_ub_ternary
    )

    return np.allclose(a=degradation_from_ode, b=degradation_from_sim, atol=0, rtol=0.001)


"""
SOLUTION FORMATTING AND VISUALIZATION
"""


def dataframe_concentrations(solve_ivp_result: sklearn.utils.Bunch, num_Ub_steps: int) -> pd.DataFrame:
    """Creates dataframe of species amounts at time points.

    Creates dataframe D from solution returned by calc_concentrations()
    where columns are species and rows are time points such that D[i,j] is the amount of species j at time i.

    Parameters
    ----------
    solve_ivp_result : sklearn.utils.Bunch
        Solution returned by calc_concentrations().

    num_Ub_steps : int
        Number of ubiquitination steps.

    Returns
    -------
    pd.DataFrame
        Columns: time and species
        Rows: species amounts at time points
    """
    result = pd.DataFrame(
        data=solve_ivp_result.y.T,
        columns=(
                [
                    'BPD_ec',
                    'BPD_ic',
                    'T',
                    'E3',
                    'BPD_T',
                    'BPD_E3',
                    'Ternary'
                ]
                + ['T_Ub_' + str(i) for i in range(1, num_Ub_steps + 1)]
                + ['BPD_T_Ub_' + str(i) for i in range(1, num_Ub_steps + 1)]
                + ['Ternary_Ub_' + str(i) for i in range(1, num_Ub_steps + 1)]
        )
    )
    result.insert(0, 't', solve_ivp_result.t)
    return result


def plot_concentrations(df):
    """
    Plot amounts of species over time.

    Args:
        df: pandas.DataFrame; returned by dataframe_concentrations().
    """
    ax = df.plot(
        x='t',
        xlabel='Time (hours)',
        ylabel='Amount (uM)',
        kind='bar',
        stacked=True,
        logy=False,
        title='Species in target protein degradation',
        figsize=(12, 8),
        fontsize=20
    )
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()


"""
UNIT TESTING
"""


def test_total_species(df: pd.DataFrame, regex: str) -> bool:
    """Tests consistency of subset of species over time.

    Parameters
    ----------
    df : pd.DataFrame
        Species amounts where columns are species and rows are time points.

    regex : str
        Regular expression for selecting columns.

    Returns
    -------
    bool
        Whether amount of subset of species is consistent over time.
    """
    totals: pd.Series[float] = df.filter(regex=regex).sum(axis=1)  # total amounts at time points
    baseline: float = totals.iloc[0]  # baseline initial value
    is_success: bool = np.allclose(a=totals, b=baseline, atol=0, rtol=0.001)
    if not is_success:
        print(totals)

    return is_success


def test_total_BPD(df: pd.DataFrame) -> bool:
    """Tests consistency of total BPD.

    Total BPD amount ({extra,intra}cellular) should remain consistent over time without pharmacokinetics.

    Parameters
    ----------
    df : pd.DataFrame
        Species amounts where columns are species and rows are time points.

    Returns
    -------
    bool
        Whether amount of total BPD is consistent over time.
    """
    return test_total_species(df, regex='(.*BPD.*)|(Ternary.*)')


def test_total_E3(df) -> bool:
    """Tests consistency of total E3.

    Total E3 amount should remain consistent over time.

    Parameters
    ----------
    df : pd.DataFrame
        Species amounts where columns are species and rows are time points.

    Returns
    -------
    bool
        Whether amount of total E3 is consistent over time.
    """
    return test_total_species(df, regex='(.*E3.*)|(Ternary.*)')


def passes_unit_tests(df: pd.DataFrame) -> bool:
    """Performs unit tests on model solution.

    Parameters
    ----------
    df : pd.DataFrame
        Species amounts where columns are species and rows are time points.


    Returns
    -------
    bool
        Whether model solution passes all unit tests.
    """
    test_results: set[str] = set()

    if not test_total_BPD(df):
        test_results.add("BPD test failed.")
    if not test_total_E3(df):
        test_results.add("E3 test failed.")

    if not test_results:
        # if test_results is empty, no tests failed
        print("All unit tests passed.")
    else:
        print(*test_results, sep='\n')

    return not test_results


def test_degradation_rate():
    """
    Calculated target protein degradation rate per hour should equal sum of
    intrinsic and UPS degradation rates.
    """
    pass
