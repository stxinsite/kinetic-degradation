"""This module contains functions that mathematically model
concentrations of species in ternary complex formation at equilibrium.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

"""
TRANSFORMED VARIABLES
"""


def equilibrium_f(variables: NDArray[float],
                  total_target: float,
                  total_protac: float,
                  total_e3: float,
                  kd_target: float,
                  kd_e3: float,
                  alpha: float) -> NDArray[float]:
    """System of equations describes concentrations at equilibrium"""
    target = np.square(variables[0])
    e3 = np.square(variables[1])
    ternary = np.square(variables[2])

    f = np.empty(3)
    f[0] = target + kd_e3 * ternary / (alpha * e3) + ternary - total_target
    f[1] = (
            kd_target * kd_e3 * ternary / (alpha * target * e3)
            + kd_e3 * ternary / (alpha * e3)
            + kd_target * ternary / (alpha * target)
            + ternary
            - total_protac
    )
    f[2] = e3 + kd_target * ternary / (alpha * target) + ternary - total_e3

    return f


def equilibrium_jac(variables: NDArray[float],
                    total_target: float,
                    total_protac: float,
                    total_e3: float,
                    kd_target: float,
                    kd_e3: float,
                    alpha: float) -> NDArray[float]:
    """Computes Jacobian of equilibrium_f()"""
    v1 = variables[0]
    v2 = variables[1]
    v3 = variables[2]
    target = np.square(v1)
    e3 = np.square(v2)
    ternary = np.square(v3)

    jac = [[2 * v1, -2 * kd_e3 * ternary / (alpha * v2 ** 3), 2 * kd_e3 * v3 / (alpha * e3) + 2 * v3],
           [-2 * kd_target * kd_e3 * ternary / (alpha * v1 ** 3 * e3) - 2 * kd_target * ternary / (alpha * v1 ** 3),
            -2 * kd_target * kd_e3 * ternary / (alpha * target * v2 ** 3) - 2 * kd_e3 * ternary / (alpha * v2 ** 3),
            2 * kd_target * kd_e3 * v3 / (alpha * target * e3)
            + 2 * kd_e3 * v3 / (alpha * e3)
            + 2 * kd_target * v3 / (alpha * target)
            + 2 * v3],
           [-2 * kd_target * ternary / (alpha * v1 ** 3), 2 * v2, 2 * kd_target * v3 / (alpha * target) + 2 * v3]]

    return jac


def noncooperative_f(total_target: float,
                     total_protac: float,
                     total_e3: float,
                     kd_target: float,
                     kd_e3: float) -> NDArray[float]:
    """Non-cooperative equilibrium system of equations"""
    target = (
            total_target
            - (total_target + total_protac + kd_target
               - np.sqrt((total_target + total_protac + kd_target) ** 2 - 4 * total_target * total_protac)) / 2
    )
    e3 = (
            total_e3
            - (total_e3 + total_protac + kd_e3
               - np.sqrt((total_e3 + total_protac + kd_e3) ** 2 - 4 * total_e3 * total_protac)) / 2
    )

    target = target if target >= 0 else 0
    e3 = e3 if e3 >= 0 else 0

    phi_ab = total_target - target if total_target > target else 0
    phi_bc = total_e3 - e3 if total_e3 > e3 else 0
    ternary = phi_ab * phi_bc / total_protac if total_protac > 0 else 0

    return np.array([target, e3, ternary])


def predict_ternary(total_target: float,
                    total_protac: float,
                    total_e3: float,
                    kd_target: float,
                    kd_e3: float,
                    alpha: float,
                    return_all: bool = False) -> Union[float, NDArray[float]]:
    noncoop_sols = noncooperative_f(total_target, total_protac, total_e3, kd_target, kd_e3)
    init_guess = np.sqrt(noncoop_sols)  # initial guesses for sqrt([target], [e3], [ternary])
    args = (total_target, total_protac, total_e3, kd_target, kd_e3, alpha)
    roots = root(equilibrium_f, init_guess, jac=equilibrium_jac, args=args, options={"maxfev": 5000})

    assert roots.success, "scipy.optimize.root() did not exit successfully"

    if return_all:
        return np.square(roots.x)  # solutions for [target], [e3], [ternary]

    return np.square(roots.x[2])  # solution for [ternary]
