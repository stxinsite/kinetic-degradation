import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import multiprocessing as mp
import kinetic_module.kinetic_tests as kinetic_tests
from kinetic_module.calc_full_config import KineticParameters

def vary_alpha(params, kon_T_binary, PROTAC_ID):
    t = [24]  # time points at which to evaluate solver
    initial_BPD_ec_conc = [0.1]  # various initial concentrations of BPD_ec (uM)

    params['kon_T_binary'] = kon_T_binary
    kon_T_binary_results = []
    alpha_range = np.geomspace(start = 0.1, stop = 300, num = 50)  # vary alpha
    for alpha in tqdm(alpha_range):
        params['alpha'] = alpha
        Params = KineticParameters(params)
        ParamsDict = Params.get_dict()

        result = kinetic_tests.solve_target_degradation(ParamsDict, t, initial_BPD_ec_conc, PROTAC_ID, save_plot=False)
        kon_T_binary_results.append(result)  # append result from using this alpha

    kon_T_binary_df = pd.concat(kon_T_binary_results)  # concat results from using this kon_T_binary
    # add alpha and kon_T_binary ID columns
    kon_T_binary_df['alpha'] = alpha_range
    kon_T_binary_df['kon_T_binary'] = kon_T_binary
    return kon_T_binary_df

if __name__ == '__main__':
    PROTAC_ID = input('PROTAC ID: ')

    with open(f"./data/{PROTAC_ID}_config.json") as file:
        params = json.load(file)  # load original config

    KP = KineticParameters(params)
    if not KP.test_closure():  # test consistency of provided parameters
        print("Kinetic parameters are not consistent")
        print(json.dumps(KP.get_dict(), indent=4))
        exit()

    # these parameters will be updated
    keys_to_update = [
        'koff_T_binary', 'koff_T_ternary', 'koff_E3_binary', 'koff_E3_ternary',
        'Kd_T_ternary', 'Kd_E3_ternary'
    ]
    for key in keys_to_update:
        params[key] = None

    pool = mp.Pool(processes=mp.cpu_count())

    # three levels of kon_T_binary
    kon_T_binary_original = params['kon_T_binary']
    kon_T_binary_small = kon_T_binary_original * 0.1
    kon_T_binary_large = kon_T_binary_original * 10
    kon_T_binaries = [kon_T_binary_original, kon_T_binary_small, kon_T_binary_large]
    inputs = [(params, kon_T_binary, PROTAC_ID) for kon_T_binary in kon_T_binaries]
    outputs = pool.starmap(vary_alpha, inputs)
    # # concat results
    all_results = pd.concat(outputs)
    all_results.to_csv(f"./saved_objects/{PROTAC_ID}_kon_vs_alpha.csv")  # save dataframe

# print(all_results)
