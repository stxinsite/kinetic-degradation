"""
This module contains functions used for a kinetic model of target protein
degradation.
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import scipy.integrate as integrate
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
        - params['PS_cell'] * params['num_cells'] * ((params['fu_ec'] * BPD_ec / params['Vec']) - (params['fu_ic'] * BPD_ic / params['Vic']))
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
        + params['kdeg_UPS'] * (0 if len(BPD_T_Ubs) == 0 else BPD_T_Ubs[-1])
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
        + params['kde_ub'] * (0 if len(T_Ubs) == 0 else T_Ubs[0])
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
        - params['koff_T_binary'] * BPD_T
        - params['kon_E3_ternary'] * BPD_T * E3 / params['Vic']
        + params['koff_E3_ternary'] * Ternary
        - params['kdeg_T'] * BPD_T
        + params['kde_ub'] * (0 if len(BPD_T_Ubs) == 0 else BPD_T_Ubs[0])
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
        + params['kdeg_UPS'] * (0 if len(Ternary_Ubs) == 0 else Ternary_Ubs[-1])
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
        - (params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['kub']) * Ternary_Ub_consec_pair[1]
        - params['kdeg_UPS'] * (Ternary_Ub_consec_pair[1] if i == params['n'] else 0)
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
    T_Ubs_rates = []
    BPD_T_Ubs_rates = []
    Ternary_Ubs_rates = []

    if params['n'] > 0:
        # there is at least one ubiquitination step
        T_all = np.append(T_Ubs, values=0)
        T_pairs = np.lib.stride_tricks.sliding_window_view(T_all, window_shape=2)

        BPD_T_all = np.append(BPD_T_Ubs, values=0)
        BPD_T_pairs = np.lib.stride_tricks.sliding_window_view(BPD_T_all, window_shape=2)

        Ternary_all = np.insert(Ternary_Ubs, obj=0, values=Ternary)  # prepend Ternary to Ternary_Ubs
        Ternary_pairs = np.lib.stride_tricks.sliding_window_view(Ternary_all, window_shape=2)

        for i in range(params['n']):
            T_Ubs_rates.append(dT_Ubdt(T_pairs[i], BPD_ic, BPD_T_Ubs[i], BPD_E3, Ternary_Ubs[i], i + 1, params))
            BPD_T_Ubs_rates.append(dBPD_T_Ubdt(BPD_T_pairs[i], BPD_ic, T_Ubs[i], E3, Ternary_Ubs[i], i + 1, params))
            Ternary_Ubs_rates.append(dTernary_Ubdt(Ternary_pairs[i], T_Ubs[i], E3, BPD_T_Ubs[i], BPD_E3, i + 1, params))

    # list concatenation
    all_rates = np.array(
        [ BPD_ec_rate, BPD_ic_rate, T_rate, E3_rate, BPD_T_rate, BPD_E3_rate, Ternary_rate ]
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

    dBPD_ecdtdy = (
        [
            -params['PS_cell'] * params['num_cells'] * params['fu_ec'] / params['Vec'],
            params['PS_cell'] * params['num_cells'] * params['fu_ic'] / params['Vic']
        ] +
        [0] * (5 + 3 * params['n'])  # dBPD_ec/dt doesn't depend on T, E3, BPD_T, BPD_E3, Ternary, T_Ubs, BPD_T_Ubs, Ternary_Ubs
    )

    dBPD_icdtdy = (
        [
            params['PS_cell'] * params['fu_ec'] / params['Vec'],
            -params['PS_cell'] * params['fu_ic'] / params['Vic'] - params['kon_T_binary'] * params['fu_ic'] * (T + np.sum(T_Ubs)) / params['Vic'] - params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
            -params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            -params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            params['koff_T_binary'] + params['kdeg_T'],
            params['koff_E3_binary'],
            0
        ] +
        ([-params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic']] * params['n']) +
        ([params['koff_T_binary'] + params['kdeg_T']] * params['n']) +
        ([0] * params['n'])  # dBPD_ic/dt does not depend on Ternary_Ubs
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
        ([params['kde_ub']] * (0 if params['n'] == 0 else 1)) + ([0] * (params['n'] - 1)) +
        ([0] * (2 * params['n']))
    )

    dE3dtdy = (
        [
            0,
            -params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
            0,
            -params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'] - params['kon_E3_ternary'] * (BPD_T + np.sum(BPD_T_Ubs)) / params['Vic'],
            -params['kon_E3_ternary'] * E3 / params['Vic'],
            params['koff_E3_binary'],
            params['koff_E3_ternary']
        ] +
        ([0] * (2 * params['n'])) +
        ([params['koff_E3_ternary']] * params['n'])
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
        ] +
        ([0] * params['n']) +
        ([params['kde_ub']] * (0 if params['n'] == 0 else 1)) + ([0] * (params['n'] - 1)) +
        ([0] * params['n'])
    )

    dBPD_E3dtdy = (
        [
            0,
            params['kon_E3_binary'] * params['fu_ic'] * E3 / params['Vic'],
            -params['kon_T_ternary'] * BPD_E3 / params['Vic'],
            params['kon_E3_binary'] * params['fu_ic'] * BPD_ic / params['Vic'],
            0,
            -params['koff_E3_binary'] - params['kon_T_ternary'] * (T + np.sum(T_Ubs)) / params['Vic'],
            params['koff_T_ternary'] + params['kdeg_T']
        ] +
        ([params['kon_T_ternary'] * BPD_E3 / params['Vic']] * params['n']) +
        ([0] * params['n']) +
        ([params['koff_T_ternary'] + params['kdeg_T']] * (params['n'] - 1)) +
        ([params['koff_T_ternary'] + params['kdeg_T'] + params['kdeg_UPS']] * (0 if params['n'] == 0 else 1))
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
        ] +
        [0] * (3 * params['n']) # dTernary/dt doesn't depend on T_Ubs, BPD_T_Ubs, Ternary_Ubs
    )

    dT_Ubdtdy_all = []  # initialize empty list for (dT_Ub/dt) / dy
    dBPD_T_Ubdtdy_all = []  # initialize empty list for (dBPD_T_Ub/dt) / dy
    dTernary_Ubdtdy_all = []  # initialize empty list for (dTernary_Ub/dt) / dy
    if params['n'] > 0:
        # there are ubiquitination steps
        T_all = np.append(T_Ubs, values=0)
        T_pairs = np.lib.stride_tricks.sliding_window_view(T_all, window_shape=2)

        BPD_T_all = np.append(BPD_T_Ubs, values=0)
        BPD_T_pairs = np.lib.stride_tricks.sliding_window_view(BPD_T_all, window_shape=2)

        Ternary_all = np.insert(Ternary_Ubs, obj=0, values=Ternary)  # prepend Ternary to Ternary_Ubs
        Ternary_pairs = np.lib.stride_tricks.sliding_window_view(Ternary_all, window_shape=2)

        for i in range(params['n']):  # for each ternary complex ubiquitination step
            dT_Ubdtdy = [0] * (7 + 3 * params['n'])
            dBPD_T_Ubdtdy = [0] * (7 + 3 * params['n'])
            dTernary_Ub_idtdy = [0] * (7 + 3 * params['n'])  # initalize list of zeros for (dTernary_Ub_i/dt) / dy

            dT_Ubdtdy[1] = -params['kon_T_binary'] * params['fu_ic'] * T_pairs[i][0] / params['Vic']  # (dT.Ubi/dt) / dBPD_ic
            dT_Ubdtdy[5] = params['kon_T_ternary'] * T_pairs[i][0] / params['Vic']  # (dT.Ubi/dt) / dBPD_E3
            dT_Ubdtdy[7+i] = -(params['kde_ub'] + params['kdeg_T']) - \
                params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic'] - \
                (params['kdeg_UPS'] if i == params['n'] else 0)  # (dT.Ubi/dt) / dT.Ubi
            dT_Ubdtdy[7+i+1] = params['kde_ub']  # (dT.Ubi/dt) / dT.Ub<i+1>
            dT_Ubdtdy[7+params['n']+i] = params['koff_T_binary']  # (dT.Ubi/dt) / dBPD.T.Ubi
            dT_Ubdtdy[7+2*params['n']+i] = params['koff_T_ternary']  # (dT.Ubi/dt) / dTernary.Ubi

            dBPD_T_Ubdtdy[1] = params['kon_T_binary'] * params['fu_ic'] * T_pairs[i][0] / params['Vic']  # (dBPD.T.Ubi/dt) / dBPD_ic
            dBPD_T_Ubdtdy[3] = -params['kon_E3_ternary'] * BPD_T_pairs[i][0] / params['Vic']  # (dBPD.T.Ubi/dt) / dE3
            dBPD_T_Ubdtdy[7+i] = params['kon_T_binary'] * params['fu_ic'] * BPD_ic / params['Vic']  # (dBPD.T.Ubi/dt) / dT.Ubi
            dBPD_T_Ubdtdy[7+params['n']+i] = -(params['kde_ub'] + params['kdeg_T']) - \
                params['koff_T_binary'] - \
                params['kon_E3_ternary'] * E3 / params['Vic'] - \
                (params['kdeg_UPS'] if i == params['n'] else 0)  # (dBPD.T.Ubi/dt) / dBPD.T.Ubi
            dBPD_T_Ubdtdy[7+params['n']+i+1] = params['kde_ub']  # (dBPD.T.Ubi/dt) / dBPD.T.Ub<i+1>
            dBPD_T_Ubdtdy[7+2*params['n']+i] = params['koff_E3_ternary']  # (dBPD.T.Ubi/dt) / dTernary.Ubi

            dTernary_Ub_idtdy[3] = params['kon_E3_ternary'] * BPD_T_pairs[i][0] / params['Vic']  # (dTernary.Ubi/dt) / dE3
            dTernary_Ub_idtdy[5] = params['kon_T_ternary'] * T_pairs[i][0] / params['Vic']  # (dTernary.Ubi/dt) / dBPD.E3
            dTernary_Ub_idtdy[7+i] = params['kon_T_ternary'] * BPD_E3 / params['Vic']  # (dTernary.Ubi/dt) / dT.Ubi
            dTernary_Ub_idtdy[7+params['n']+i] = params['kon_E3_ternary'] * E3 / params['Vic']  # (dTernary.Ubi/dt) / dBPD.T.Ubi
            dTernary_Ub_idtdy[7+2*params['n']+i-1] = params['kub']  # (dTernary_Ub_i/dt) / dTernary_Ub_<i-1>
            dTernary_Ub_idtdy[7+2*params['n']+i] = -(params['kdeg_T'] + params['koff_T_ternary'] + params['koff_E3_ternary'] + params['kub']) - \
                (params['kdeg_UPS'] if i == params['n'] else 0)  # (dTernary_Ub_i/dt) / dTernary_Ub_i

            dT_Ubdtdy_all.append(dT_Ubdtdy)
            dBPD_T_Ubdtdy_all.append(dBPD_T_Ubdtdy)
            dTernary_Ubdtdy_all.append(dTernary_Ub_idtdy)

    # (7 + 3*n) x (7 + 3*n) array
    all_jacs = np.array(
        [dBPD_ecdtdy, dBPD_icdtdy, dTargetdtdy, dE3dtdy, dBPD_Tdtdy, dBPD_E3dtdy, dTernarydtdy]  # 7 x (7 + 3*n) list of lists
        + dT_Ubdtdy_all  # n x (7 + 3*n) list of lists
        + dBPD_T_Ubdtdy_all # n x (7 + 3*n) list of lists
        + dTernary_Ubdtdy_all # n x (7 + 3*n) list of lists
    )

    return all_jacs


"""
SOLVE SYSTEM OF ODES
"""


def initial_values(params, BPD_ec=0, BPD_ic=0):
    """Returns array of initial values for species amounts.

    Args:
        params: dict; model parameters.
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

def calc_concentrations(t_eval, y0, params, max_step=np.inf, T_total_baseline=None, T_total_steady_state=None):
    """
    Solve the initial value problem for the amounts of species in
    PROTAC-induced target protein degradation via the ubiquitin-proteasome system.

    Args:
        t_eval: array_like; time points at which to store computed solution.
        y0: array_like; the initial values of all species in the system in the following order:
            [
                BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary,
                T_Ub_1, ..., T_Ub_n, BPD_T_Ub_1, ..., BPD_T_Ub_n, Ternary_Ub_1, ..., Ternary_Ub_n
            ].
        params: dict; kinetic rate constants and model parameters for rate equations.
        max_step: float; maximum allowed step size for solver. It is recommended to set
            to small value (e.g. 0.001) for systems with small initial values.
        T_total_baseline: total amount of Target in system at baseline.
        T_total_steady_state: total amount of Target at system steady state.
    """
    def rates(t, y, params, T_total_baseline, T_total_steady_state):
        """
        Return ODEs evaluated at (t, y).
        Must have call signature func(t, y, *args).
        """
        return kinetic_rates(y, params)

    def jac_rates(t, y, params, T_total_baseline, T_total_steady_state):
        """
        Return Jacobian of ODEs evaluated at (t, y).
        Must have identical call signature as rates().
        """
        return jac_kinetic_rates(y, params)

    def t_half_event(t, y, params, T_total_baseline, T_total_steady_state):
        """
        Return difference between total amount of Target in system at t and
        halfway to Target steady state from baseline.
        Must have identical call signature as rates().
        """
        C = 0.5 * (T_total_baseline - T_total_steady_state)  # halfway between baseline and steady state
        T_total_at_t = np.sum(np.concatenate((y[[2,4]], y[6:])))  # total amount of Target in system at t
        return T_total_at_t - C

    t_half_event.terminal = True  # if t_half_event changes sign, solver will terminate
    # tmin = np.min(times)
    tmax = np.max(t_eval)

    result = integrate.solve_ivp(rates, (0, tmax), y0,
                                 method = 'BDF',
                                 t_eval = t_eval,
                                 args = (params, T_total_baseline, T_total_steady_state),
                                 max_step = max_step,
                                 jac = jac_rates,
                                 events = t_half_event if (T_total_baseline is not None) and (T_total_steady_state is not None) else None
                                 )

    if not np.all(result.y >= 0):
        print("Solution contains negative values at some time points. Trying smaller max_step size.")
        max_step = max_step / 2  # halve the max_step
        return calc_concentrations(t_eval, y0, params, max_step, T_total_baseline, T_total_steady_state)

    return result

def solve_steady_state(initial_guess, params: dict[str, float]):
    """
    Solve system steady state, the amounts of species for which all kinetic rates
    equal 0.

    Args:
        initial_guess: array; initial guess for solution
        params: dict; kinetic rate constants and model parameters for rate equations.

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
            kinetic_rates,
            initial_guess,
            jac = jac_kinetic_rates,
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

def calc_Dmax(t_eval, y0, params, initial_guess = None) -> float:
    """
    Calculate Dmax, the maximal percent target protein degradation.
    """
    if initial_guess is None:
        print("Solving IVP because no initial guess provided.")
        result = calc_concentrations(t_eval, y0, params, max_step = 0.001)
        initial_guess = result.y[:,-1]  # system state at final time point

    steady_state = solve_steady_state(initial_guess, params)
    T_total_steady_state = np.sum(np.concatenate((steady_state[[2,4]], steady_state[6:])))
    T_total_baseline = np.sum(np.concatenate((y0[[2,4]], y0[6:])))
    Dmax = 1 - T_total_steady_state / T_total_baseline
    return Dmax

def calc_t_half(t_eval, y0, params):
    """
    Find time t at which total amount of Target reaches halfway between baseline and
    steady state.
    """
    result = calc_concentrations(t_eval=t_eval, y0=y0, params=params, max_step = 0.001)
    init_guess = result.y[:,-1]  # system state at the last time point
    steady_state = solve_steady_state(initial_guess=init_guess, params=params)
    T_total_steady_state = np.sum(np.concatenate((steady_state[[2,4]], steady_state[6:])))
    T_total_baseline = np.sum(np.concatenate((y0[[2,4]], y0[6:])))

    result_with_events = calc_concentrations(
        t_eval=t_eval, y0=y0, params=params, max_step = 0.001,
        T_total_baseline=T_total_baseline, T_total_steady_state=T_total_steady_state
    )

    t_half = result_with_events.t_events[0][0]  # first event type, first time point
    return t_half

def calc_degradation_curve(t_eval, params, initial_BPD_ec_conc=0, initial_BPD_ic_conc=0, return_only_final_state=True):
        """
        Calculate target protein degradation and ternary formation
        for fixed initial degrader concentration at time points t_eval.

        Arguments:
            t_eval: array_like; time points at which to store computed solution.
            params: dict; kinetic rate constants and model parameters for rate equations.
            initial_BPD_ec_conc: float; initial value of BPD_ec concentration.
            initial_BPD_ic_conc: float; initial value of BPD_ic concentration.
            return_only_final_state: bool; whether to return only final state of system.

        Returns:
            pd.DataFrame; percent degradation and ternary formation relative to baseline Target at time points t_eval.
        """
        initial_BPD_ec = initial_BPD_ec_conc * params['Vec']
        initial_BPD_ic = initial_BPD_ic_conc * params['Vic']
        y0 = initial_values(params, BPD_ec=initial_BPD_ec, BPD_ic=initial_BPD_ic)

        # solve system of ODEs
        concentrations = calc_concentrations(t_eval, y0, params, max_step = 0.001)
        print(concentrations.success)
        concentrations_df = dataframe_concentrations(concentrations, num_Ub_steps=params['n'])  # rows are time points, columns are species

        # run unit tests
        run_unit_tests(concentrations_df)

        # calculate target protein degradation and ternary complex formation
        T_total_baseline = np.sum(np.concatenate((y0[[2,4]], y0[6:])))  # float: total amount of Target at baseline
        T_totals = concentrations_df.filter(regex='.*T.*').sum(axis=1)  # pd.Series: total amounts of Target at time points
        Ternary_totals = concentrations_df['Ternary']  # pd.Series: amounts of un-ubiquitinated Ternary at time points
        all_Ternary_totals = concentrations_df.filter(regex='Ternary.*').sum(axis=1)  # pd.Series: total amounts of all Ternary at time points

        relative_T = T_totals / T_total_baseline * 100  # percent total Target relative to baseline Target
        relative_Ternary = Ternary_totals / T_total_baseline * 100  # percent naked Ternary relative to baseline Target
        relative_all_Ternary = all_Ternary_totals / T_total_baseline * 100  # percent total Ternary relative to baseline Target

        # calculate Dmax
        # average_relative_T = (relative_T.min() + relative_T.max()) / 2  # average of min and max Target degradation seen so far
        # relative_T_index = pd.Index(relative_T)  # index object
        # # let initial guess for steady state be system near half Target degradation
        # initial_guess_idx = relative_T_index.get_loc(average_relative_T, method = 'nearest')
        # x0 = concentrations.y[:, initial_guess_idx]
        Dmax = calc_Dmax(t_eval, y0, params, initial_guess=concentrations.y[:,-1]) * 100  # float: percent Dmax

        result = pd.DataFrame({
            'initial_BPD_ec_conc': initial_BPD_ec_conc,
            'initial_BPD_ic_conc': initial_BPD_ic_conc,
            'degradation': relative_T,
            'Ternary': relative_Ternary,
            'all_Ternary': relative_all_Ternary,
            'Dmax': Dmax
        })
        result.insert(0, 't', pd.Series(t_eval))

        if return_only_final_state:
            return result.iloc[-1:]  # only return system at last time point

        return result

"""
RESULT MANIPULATION AND VISUALIZATION
"""
def dataframe_concentrations(solve_ivp_result, num_Ub_steps):
    """
    Create pandas.DataFrame from object returned by scipy.integrate.solve_ivp()
    where D[i,j] is amount of species j at time i.

    Args:
        solve_ivp_result: OptimizeResult; returned by scipy.integrate.solve_ivp().
        num_Ub_steps: int; number of ubiquitination steps.
    """
    result = pd.DataFrame(
        solve_ivp_result.y.T,
        columns=(
            [
                'BPD_ec',
                'BPD_ic',
                'T',
                'E3',
                'BPD_T',
                'BPD_E3',
                'Ternary'
            ] +
            ['T_Ub_' + str(i) for i in range(1, num_Ub_steps+1)] +
            ['BPD_T_Ub_' + str(i) for i in range(1, num_Ub_steps+1)] +
            ['Ternary_Ub_' + str(i) for i in range(1, num_Ub_steps+1)]
        )
    )

    result['t'] = solve_ivp_result.t  # add column `t` for time points
    return result

def plot_concentrations(df):
    """
    Plot amounts of species over time.

    Args:
        df: pandas.DataFrame; returned by dataframe_concentrations().
    """
    ax = df.plot(x='t',
                 xlabel='Time (hours)',
                 ylabel='Amount (uM)',
                 kind='bar',
                 stacked=True,
                 logy=False,
                 title='Species in target protein degradation',
                 figsize = (12, 8),
                 fontsize = 20
                 )
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.show()

"""
UNIT TESTING
"""


def test_total_species(df, regex):
    totals = df.filter(regex=regex).sum(axis=1)  # pd.Series: total amounts at time points
    baseline = totals.iloc[0]
    is_success = np.allclose(totals, baseline)
    if not is_success:
        print(totals)

    return is_success


def test_total_BPD(df):
    """
    Total BPD amount should remain constant over time without in vivo PK dynamics.
    """
    return test_total_species(df, regex='.*BPD.*')


def test_total_E3(df):
    """
    Total E3 amount should remain constant over time.
    """
    return test_total_species(df, regex='.*E3.*')


def test_all_nonnegative(df):
    pass


def run_unit_tests(df):
    all_tests_passed = True
    if not test_total_BPD(df):
        print("BPD test failed.")
        all_tests_passed = False
    if not test_total_E3(df):
        print("E3 test failed.")
        all_tests_passed = False

    if all_tests_passed:
        print("All unit tests passed.")


def test_degradation_rate():
    """
    Calculated target protein degradation rate per hour should equal sum of
    intrinsic and UPS degradation rates.
    """
    pass
