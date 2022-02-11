"""
The effects on degradation of target proteasomal degradation rate, and the dependence on cooperativity
at different ubiquitination rates.
"""
import numpy as np
import pandas as pd
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_filename = 'SiTX_38406_config.yml'
    protac_id = 'ACBI1'
    t_eval = 6  # time point at which to calculate

    test_id = protac_id.replace(" ", "") + f"t={t_eval}"

    alpha_range = np.geomspace(start=0.1, stop=1000, num=50)  # geometric range of alpha values
    kdeg_ups_range = np.array([10, 100, 500, 1000])  # range of kdeg_UPS values

    initial_BPD_ec_conc = 0.0005  # initial concentrations of BPD_ec (uM)

    result = kt.kdeg_ups_vs_alpha(
        config_filename=config_filename,
        protac_id=protac_id,
        t_eval=t_eval,
        alpha_range=alpha_range,
        kdeg_UPS_range=kdeg_ups_range,
        initial_BPD_ec_conc=initial_BPD_ec_conc
    )
    result.to_csv(f"./saved_objects/{test_id}_kdeg_UPS_vs_alpha_DEG.csv", index=False)  # save dataframe
