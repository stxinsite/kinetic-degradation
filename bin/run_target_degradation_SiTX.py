import numpy as np
import yaml
import kinetic_module.kinetic_tests as kinetic_tests

"""
LOAD PARAMETERS FROM CONFIG
"""
with open('./data/SiTX_38404_config.yml', 'r') as file:
    params_SiTX_38404 = yaml.safe_load(file)

with open('./data/SiTX_38406_config.yml', 'r') as file:
    params_SiTX_38406 = yaml.safe_load(file)

"""
RUN TEST(S)
"""
t = [24]  # time points at which to evaluate solver
initial_BPD_ec_conc = np.logspace(base = 10.0, start = -1, stop = 5, num = 50) / 1000  # various initial concentrations of BPD_ec (uM)

kinetic_tests.solve_target_degradation(params_SiTX_38404, t, initial_BPD_ec_conc, 'SiTX_38404')
kinetic_tests.solve_target_degradation(params_SiTX_38406, t, initial_BPD_ec_conc, 'SiTX_38406')

t = np.arange(start = 0, stop = 168 + 1, step = 3)  # time points at which to evaluate solver
initial_BPD_ec_conc = [0.1]  # initial concentration of BPD_ec (uM)

kinetic_tests.solve_target_degradation(params_SiTX_38404, t, initial_BPD_ec_conc, 'SiTX_38404')
kinetic_tests.solve_target_degradation(params_SiTX_38406, t, initial_BPD_ec_conc, 'SiTX_38406')
