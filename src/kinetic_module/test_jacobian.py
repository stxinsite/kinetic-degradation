import unittest
import yaml
import numpy as np
from numpy.random import uniform
import numdifftools as nd
from kinetic_functions import kinetic_rates, jac_kinetic_rates, initial_values, calc_concentrations
from calc_full_config import KineticParameters

class MyTestCase(unittest.TestCase):

    def test_jacobian(self):
        with open(file=f'./data/unit_test_config.yml', mode='r') as file:
            config_dict = yaml.safe_load(file)

        params = KineticParameters(config_dict)
        if params.is_fully_defined():
            params = params.params
        else:
            raise ValueError('Config file provided is insufficient or inconsistent.')


        # solve system of ODEs
        t_eval = np.linspace(0, 1)
        y0 = initial_values(params, BPD_ec=0, BPD_ic=0.1 * params['Vic'])
        concentrations = calc_concentrations(t_eval, y0, params, max_step=0.001)
        y = concentrations.y[:,-1]
        jac = nd.Jacobian(kinetic_rates)(y, params)
        my_jac = jac_kinetic_rates(y, params)
        grad_err = jac - my_jac

        print(jac)
        self.assertTrue(np.allclose(grad_err, 0))


if __name__ == '__main__':
    unittest.main()
