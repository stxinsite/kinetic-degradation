import numpy as np
import yaml
import kinetic_module.kinetic_tests as kinetic_tests

"""
This script reproduces Bartlett et al. (2013) Supplementary Figures 1b and 1c.

Given initial values, we solve for target protein degradation relative to initial baseline amount
over time at various concentrations of PROTAC.
"""

"""
LOAD PARAMETERS FROM CONFIG
"""
with open('./data/degradation_config.yml', 'r') as file:
    params = yaml.safe_load(file)

"""
RUN TEST(S)
"""
t = [24]
initial_BPD_ec_conc = np.logspace(base = 10.0, start = -1, stop = 5, num = 50) / 1000  # various initial concentrations of BPD_ec (uM)

kinetic_tests.solve_target_degradation(params, t, initial_BPD_ec_conc, 'BTK')


# t = np.arange(start = 0, stop = 48 + 1, step = 2)  # time points at which to evaluate solver
# initial_BPD_ec_conc = np.logspace(base = 10.0, start = -1, stop = 5, num = 50) / 1000  # various initial concentrations of BPD_ec (uM)
