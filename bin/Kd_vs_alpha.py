import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import multiprocessing as mp
import kinetic_module.kinetic_tests as kinetic_tests
from kinetic_module.calc_full_config import KineticParameters, MyPool

def test_params_consistency(params):
    Params = KineticParameters(params)
    return Params.test_closure()

def update_params(params, keys, values):
    assert len(keys) == len(values), "Length of keys to update in params must equal length of values."
    for key_value_pair in zip(keys, values):
        params[key_value_pair[0]] = key_value_pair[1]
    Params = KineticParameters(params)
    return Params.get_dict()

def vary_alpha(params, key, value, PROTAC_ID):
    """
    Solve degradation curve at level params[key] = value across range of alpha.
    """

    params[key] = value
    results = []
    alpha_range = np.geomspace(start = 0.1, stop = 300, num = 50)  # vary alpha
    for alpha in tqdm(alpha_range):
        params['alpha'] = alpha
        Params = KineticParameters(params)
        ParamsDict = Params.get_dict()

        result = kinetic_tests.solve_target_degradation(ParamsDict, t, initial_BPD_ec_conc, PROTAC_ID, save_plot=False)
        results.append(result)  # append result from using this alpha

    results_df = pd.concat(results)  # concat results from using this value
    # add alpha and value ID columns
    results_df['alpha'] = alpha_range
    results_df[key] = value
    return results_df

if __name__ == '__main__':
#     PROTAC_ID = input('PROTAC ID: ')
    PROTAC_ID = 'SiTX_38406'

    with open(f"./data/{PROTAC_ID}_config.json") as file:
        params = json.load(file)  # load original config

    if not test_params_consistency(params):
        print("Kinetic parameters are not consistent")
        exit()

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

    num_alpha = 10
    alpha_range = np.geomspace(start = 0.1, stop = 300, num = num_alpha)  # geometric range of alpha values
    Kd_T_binary = params['Kd_T_binary']
    Kd_T_binary_range = [Kd_T_binary * 0.1, Kd_T_binary, Kd_T_binary * 10]  # range of three Kd_T_binary values

    params_copies = [params.copy()] * num_alpha  # copies of original params
    new_params = [
        update_params(params_copy, ['alpha', 'Kd_T_binary'], [alpha, Kd])
        for Kd in Kd_T_binary_range
        for (params_copy, alpha) in zip(params_copies, alpha_range)
    ]

    initial_BPD_ec_concs = [0.1]  # initial concentrations of BPD_ec (uM)
    t = [24]  # time points at which to calculate

    pool = mp.Pool(processes=mp.cpu_count())
    inputs = [(initial_BPD_ec_concs, t, params, PROTAC_ID) for params in new_params]
    outputs = pool.starmap(kinetic_tests.solve_target_degradation, inputs)

    result = pd.concat(outputs)  # concat results
    result.to_csv(f"./saved_objects/{PROTAC_ID}_Kd_vs_alpha.csv")  # save dataframe

# if __name__ == '__main__':
#     PROTAC_ID = input('PROTAC ID: ')
#
#     with open(f"./data/{PROTAC_ID}_config.json") as file:
#         params = json.load(file)  # load original config
#
#     KP = KineticParameters(params)
#     if not KP.test_closure():  # test consistency of provided parameters
#         print("Kinetic parameters are not consistent")
#         print(json.dumps(KP.get_dict(), indent=4))
#         exit()
#
#     # these parameters will be updated
#     keys_to_update = [
#         'koff_T_binary', 'koff_T_ternary', 'koff_E3_ternary',
#         'Kd_T_ternary', 'Kd_E3_ternary'
#     ]
#     for key in keys_to_update:
#         params[key] = None
#
#     pool = mp.Pool(processes=mp.cpu_count())
#
#     # three levels of Kd_T_binary
#     original = params['Kd_T_binary']
#     small = original * 0.1
#     large = original * 10
#     values = [original, small, large]
#     inputs = [(params, 'Kd_T_binary', value, PROTAC_ID) for value in values]
#     outputs = pool.starmap(vary_alpha, inputs)
#     # # concat results
#     all_results = pd.concat(outputs)
#     all_results.to_csv(f"./saved_objects/{PROTAC_ID}_Kd_vs_alpha.csv")  # save dataframe

# print(all_results)
