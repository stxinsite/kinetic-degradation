"""
The effects on degradation of ubiquitination rate, and the dependence on cooperativity
at different ubiquitination rates.
"""
import numpy as np
import pandas as pd
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_filename = 'SiTX_38406_config.yml'
    protac_id = 'ACBI1'

    t_eval = 6  # time point at which to calculate
    initial_BPD_ec_conc = 0.001  # initial concentrations of BPD_ec (uM)

    test_id = protac_id.replace(" ", "") + f"_bpd_ec={initial_BPD_ec_conc}_t={t_eval}"

    alpha_range = np.geomspace(start=0.1, stop=1000, num=50)  # geometric range of alpha values
    kub_range = np.array([10, 100, 1000, 10000])  # range of kub values

    result = kt.kub_vs_alpha(
        config_filename=config_filename,
        protac_id=protac_id,
        t_eval=t_eval,
        alpha_range=alpha_range,
        kub_range=kub_range,
        initial_BPD_ec_conc=initial_BPD_ec_conc
    )
    result.to_csv(f"./saved_objects/{test_id}_kub_vs_alpha.csv", index=False)  # save dataframe
