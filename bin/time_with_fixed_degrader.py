import numpy as np
import pandas as pd
import yaml
import multiprocessing as mp
import kinetic_module.kinetic_tests as kinetic_tests
from kinetic_module.calc_full_config import KineticParameters

if __name__ == '__main__':
        """
        LOAD PARAMETERS FROM CONFIG
        """
    with open('./data/SiTX_38404_config.yml', 'r') as file:
        params_38404 = yaml.safe_load(file)

    with open('./data/SiTX_38406_config.yml', 'r') as file:
        params_38406 = yaml.safe_load(file)

    Params_38404 = KineticParameters(params_38404)
    Params_38406 = KineticParameters(params_38406)

    if not (Params_38404.is_fully_defined() and Params_38406.is_fully_defined()):
        print("Kinetic parameters are not consistent")
        exit()
    else:
        print("Kinetic parameters are consistent")
        full_params_38404 = Params_38404.get_dict()
        full_params_38406 = Params_38406.get_dict()

    all_params = [full_params_38404, full_params_38406]
    """
    RUN TEST(S)
    """
    initial_BPD_ec_concs = [0.1]  # initial concentration of BPD_ec (uM)
    t = np.linspace(0, 12, num=50)  # time points at which to evaluate solver

    pool = mp.Pool(processes=mp.cpu_count())
    inputs = [(initial_BPD_ec_concs, t, params, False) for params in all_params]
    outputs = pool.starmap(kinetic_tests.solve_target_degradation, inputs)
    pool.close()
    pool.join()

    result = pd.concat(outputs)
    result['PROTAC'] = np.repeat(np.array(['PROTAC 1', 'ACBI1']), repeats = len(initial_BPD_ec_concs) * len(t))
    result.to_csv("./saved_objects/SiTX_38404+38406_DEG.csv")
