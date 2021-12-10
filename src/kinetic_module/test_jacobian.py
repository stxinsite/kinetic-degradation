"""Unit tests for Jacobian of rate equations.
"""
import unittest
import numpy as np
from numpy.random import random
from scipy.optimize import check_grad
import numdifftools as nd
import kinetic_tests as kt
from kinetic_functions import kinetic_rates, jac_kinetic_rates, initial_values, calc_concentrations


def wrap_kinetic_rates(y, params, i):
    result = kinetic_rates(y, params)
    return result[i]


def wrap_jac_kinetic_rates(y, params, i):
    result = jac_kinetic_rates(y, params)
    return result[i, :]


def wrap_check_grad(x0, params, n_species):
    grad_err = np.empty(n_species, dtype=np.float64)
    for i in range(n_species):
        grad_err[i] = check_grad(wrap_kinetic_rates, wrap_jac_kinetic_rates, x0, params, i)

    return grad_err


class MyTestCase(unittest.TestCase):
    def test_jacobian(self):
        params = kt.get_params_from_config('unit_test_config.yml')

        y0 = initial_values(params, BPD_ec=0.1*params['Vec'], BPD_ic=0.1*params['Vec'])
        n_species = len(y0)

        grad_err = wrap_check_grad(y0, params, n_species)
        print(grad_err)
        self.assertTrue(np.allclose(grad_err, 0))

    def test_jacobian_at_time(self):
        params = kt.get_params_from_config('unit_test_config.yml')

        y0 = initial_values(params, BPD_ec=0.1*params['Vec'], BPD_ic=0.1*params['Vec'])
        n_species = len(y0)

        result = calc_concentrations(t_eval=np.linspace(0,1), y0=y0, params=params, max_step=0.001)
        y_states = result.y
        x0 = y_states[:,-1]

        grad_err = wrap_check_grad(x0, params, n_species)
        print(grad_err)
        self.assertTrue(np.allclose(grad_err, 0))

    def test_jacobian_with_nd(self):
        params = kt.get_params_from_config('unit_test_config.yml')

        y0 = initial_values(params, BPD_ec=0.1*params['Vec'], BPD_ic=0.1*params['Vec'])
        n_species = len(y0)

        result = calc_concentrations(t_eval=np.linspace(0,1), y0=y0, params=params, max_step=0.001)
        y_states = result.y
        x0 = y_states[:,-1]

        nd_jac = nd.Jacobian(kinetic_rates)
        grad_diff = nd_jac(x0, params) - jac_kinetic_rates(x0, params)
        grad_err = np.sqrt(np.sum(np.square(grad_diff), axis=1))
        print(grad_err)
        self.assertTrue(np.allclose(grad_err, 0, atol=0.0005))


if __name__ == '__main__':
    unittest.main()
