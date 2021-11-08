import yaml
import kinetic_module.ternary_formation_test as tf_test

"""
LOAD PARAMETERS FROM CONFIG
"""
with open('./data/ternary_formation_config.yml', 'r') as file:
    params = yaml.safe_load(file)

"""
RUN TEST(S)
"""
tf_test.solve_ODE(params)
