import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
import kinetic_module.kinetic_tests as kinetic_tests
from kinetic_module.calc_full_config import KineticParameters

if __name__ == '__main__':
    PROTAC_ID = 'SiTX_38406'

    with open(f"./data/{PROTAC_ID}_config.yml") as file:
        params = yaml.safe_load(file)  # load original config

    Params = KineticParameters(params)

    if not Params.is_fully_defined():
        print("Kinetic parameters are not consistent")
        exit()
    else:
        print("Kinetic parameters are consistent")
        full_params = Params.get_dict()

    PS_cell = full_params['PS_cell']
    factors = np.power(10, np.arange(-2, 3, dtype = float))
    # factors = np.power(10, np.arange(0, 2, dtype=float))
    PS_cell_range = PS_cell * factors
    # print(PS_cell_range)
    new_params = []
    for PS in PS_cell_range:
        params_copy = full_params.copy()
        params_copy['PS_cell'] = PS
        new_params.append(params_copy)

    initial_BPD_ec_concs = [0.1]  # initial concentrations of BPD_ec (uM)
    # t = np.arange(0, 17)  # time points at which to calculate
    t = np.linspace(0, 18, num=30)

    pool = mp.Pool(processes=mp.cpu_count())
    inputs = [(initial_BPD_ec_concs, t, params) for params in new_params]
    # outputs = [kinetic_tests.solve_target_degradation(*i) for i in inputs]
    outputs = pool.starmap(kinetic_tests.solve_target_degradation, inputs)
    pool.close()
    pool.join()

    result = pd.concat(outputs)  # concat outputs into one pd.DataFrame
    result['PS_cell'] = np.repeat(PS_cell_range, repeats = len(initial_BPD_ec_concs) * len(t))
    result.to_csv(f"./saved_objects/{PROTAC_ID}_Permeability_DEG.csv")  # save dataframe
