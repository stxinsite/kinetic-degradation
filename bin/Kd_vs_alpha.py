import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import multiprocessing as mp
import kinetic_module.kinetic_tests as kinetic_tests
from kinetic_module.calc_full_config import KineticParameters

def vary_alpha(params, key, value, PROTAC_ID):
    """
    Solve degradation curve at level params[key] = value across range of alpha.
    """
    t = [24]  # time points at which to evaluate solver
    initial_BPD_ec_conc = [0.1]  # various initial concentrations of BPD_ec (uM)

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
        'koff_T_binary', 'koff_T_ternary', 'koff_E3_ternary',
        'Kd_T_ternary', 'Kd_E3_ternary'
    ]
    for key in keys_to_update:
        params[key] = None

    pool = mp.Pool(processes=mp.cpu_count())

    # three levels of Kd_T_binary
    original = params['Kd_T_binary']
    small = original * 0.1
    large = original * 10
    values = [original, small, large]
    inputs = [(params, 'Kd_T_binary', value, PROTAC_ID) for value in values]
    outputs = pool.starmap(vary_alpha, inputs)
    # # concat results
    all_results = pd.concat(outputs)
    all_results.to_csv(f"./saved_objects/{PROTAC_ID}_Kd_vs_alpha.csv")  # save dataframe

# print(all_results)
