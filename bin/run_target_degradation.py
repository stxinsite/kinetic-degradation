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
kinetic_tests.solve_target_degradation(params)
