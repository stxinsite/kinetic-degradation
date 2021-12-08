"""Degradation over time for fixed degrader concentration

This script calculates target protein degradation, ternary complex formation, and
Dmax for a fixed initial concentration of degrader.

YAML config file(s) must be saved to `./data/` folder.


"""

import yaml
import numpy as np
import pandas as pd
import multiprocessing as mp
import kinetic_module.kinetic_tests as kt
from kinetic_module.calc_full_config import KineticParameters


config_files = ['SiTX_38404_config.yml', 'SiTX_38406_config.yml']
protac_IDs = ['PROTAC 1', 'ACBI1']
test_ID = '&'.join(protac_IDs)
test_ID = test_ID.replace(" ", "")

t_eval = np.linspace(0, 24, num=240)
initial_BPD_ic_concs = [0.1]

outputs = []
for config, protac in zip(config_files, protac_IDs):
    with open(file=f'./data/{config}', mode='r') as file:
        config_dict = yaml.safe_load(file)

    params = KineticParameters(config_dict)
    if params.is_fully_defined():
        params = params.params

    print(params)

    df = kt.solve_target_degradation(t_eval, params,
        initial_BPD_ic_concs=initial_BPD_ic_concs,
        return_only_final_state=False,
        PROTAC_ID=protac
        )
    outputs.append(df)

result = pd.concat(outputs)
result.to_csv(f"./saved_objects/{test_ID}_DEG.csv")

# initial_BPD_ec_conc = np.logspace(base = 10.0, start = -1, stop = 5, num = 50) / 1000  # various initial concentrations of BPD_ec (uM)
# initial_BPD_ec_conc = np.logspace(base = 10.0, start = -1, stop = 5, num = 50) / 1000  # various initial concentrations of BPD_ec (uM)


# t = np.arange(start = 0, stop = 48 + 1, step = 2)  # time points at which to evaluate solver
