import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
import kinetic_module.kinetic_tests as kinetic_tests
from kinetic_module.calc_full_config import KineticParameters

def update_params(params, keys, values):
    assert len(keys) == len(values), "Length of keys to update in params must equal length of values."
    for key_value_pair in zip(keys, values):
        params[key_value_pair[0]] = key_value_pair[1]
    Params = KineticParameters(params)
    return Params.get_dict()

if __name__ == '__main__':
    PROTAC_ID = 'SiTX_38406'

    with open(f"./data/{PROTAC_ID}_config.yml", "r") as file:
        params = yaml.safe_load(file)  # load original config

    Params = KineticParameters(params)

    if not Params.is_fully_defined():
        print("Kinetic parameters are not consistent")
        exit()
    else:
        print("Kinetic parameters are consistent")
        params = Params.get_dict()

    # set these parameters to None to be updated
    keys_to_update = [
        'koff_T_binary',
        'koff_T_ternary',
        'koff_E3_ternary',
        'Kd_T_ternary',
        'Kd_E3_ternary'
    ]
    for key in keys_to_update:
        params[key] = None

    num_alpha = 40
    alpha_range = np.geomspace(start = 0.1, stop = 300, num = num_alpha)  # geometric range of alpha values
    Kd_T_binary_range = np.power(10, np.arange(-1, 3, dtype = float))  # range of Kd_T_binary values

    params_range = [(alpha, Kd) for Kd in Kd_T_binary_range for alpha in alpha_range]  # all combinations of alpha and Kd
    params_copies = [params.copy()] * len(params_range)  # copies of original params
    new_params = [
        update_params(params_copy, ['alpha', 'Kd_T_binary'], new_values)
        for (params_copy, new_values) in zip (params_copies, params_range)
    ]

    initial_BPD_ec_concs = [0.1]  # initial concentrations of BPD_ec (uM)
    t = [6]  # time points at which to calculate

    pool = mp.Pool(processes=mp.cpu_count())
    inputs = [(initial_BPD_ec_concs, t, params) for params in new_params]
    outputs = pool.starmap(kinetic_tests.solve_target_degradation, inputs)
    pool.close()
    pool.join()

    result = pd.concat(outputs)  # concat outputs into one pd.DataFrame
    result['alpha'] = np.tile(alpha_range, reps = len(initial_BPD_ec_concs) * len(t) * len(Kd_T_binary_range))
    result['Kd_T_binary'] = np.repeat(Kd_T_binary_range, repeats = len(initial_BPD_ec_concs) * len(t) * len(alpha_range))
    result.to_csv(f"./saved_objects/{PROTAC_ID}_Kd_vs_alpha_DEG.csv")  # save dataframe
