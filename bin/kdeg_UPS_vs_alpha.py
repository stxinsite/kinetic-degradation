"""
The effects on degradation of target proteasomal degradation rate, and the dependence on cooperativity
at different ubiquitination rates.
"""
from ast import arg
import os
import argparse

import numpy as np
import pandas as pd

import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulate target protein degradation across cooperativity values.')
    parser.add_argument('--config', dest='config_filename',
                        type=str, help='Name of the config file', required=True)
    parser.add_argument('--protac', dest='protac_id', type=str,
                        help='PROTAC identifier', required=True)
    parser.add_argument('-t', dest='t_eval', type=float,
                        help='Time until which to simulate', required=True)

    args = parser.parse_args()

    assert os.path.isfile(
        f'./data/{args.config_filename}'), f"{args.config_filename} does not exist"

    t_eval = int(args.t_eval) if args.t_eval.is_integer() else args.t_eval  # time point at which to calculate
    initial_bpd_ec_conc = 0.001  # initial concentrations of BPD_ec (uM)

    test_id = args.protac_id.replace(" ", "") + f"_bpd_ec={initial_bpd_ec_conc}_t={t_eval}"

    alpha_range = np.geomspace(start=0.1, stop=1000., num=50)  # geometric range of alpha values
    kdeg_ups_range = np.array([30, 100, 100])  # range of kdeg_UPS values

    result = kt.kdeg_ups_vs_alpha(
        config_filename=args.config_filename,
        protac_id=args.protac_id,
        t_eval=t_eval,
        alpha_range=alpha_range,
        kdeg_ups_range=kdeg_ups_range,
        initial_bpd_ec_conc=initial_bpd_ec_conc
    )

    # save dataframe
    result.to_csv(f"./saved_objects/{test_id}_deg_vs_kdeg.csv", index=False)
