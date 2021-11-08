import yaml
import kinetic_module.kinetic_tests as kinetic_tests

"""
This script reproduces Bartlett et al. (2013) Supplementary Figure 1a.

Given initial values, we solve for amounts of Ternary complex formed after 24 hours
at various concentrations of PROTAC.

Ternary complex formation with intracellular PROTAC only.
No ubiquitination or degradation.
"""

"""
LOAD PARAMETERS FROM CONFIG
"""
with open('./data/ternary_formation_config.yml', 'r') as file:
    params = yaml.safe_load(file)

"""
RUN TEST(S)
"""
kinetic_tests.solve_ternary_formation(params)
