import numpy as np
import pandas as pd
import kinetic_module.kinetic_tests as kt


if __name__ == '__main__':
    config_filename = 'SiTX_38406_config.yml'
    protac_id = 'ACBI1'

    t_eval = 6  # time point at which to calculate
    initial_BPD_ec_conc = 0.001  # initial concentrations of BPD_ec (uM)

    test_id = protac_id.replace(" ", "") + f"_bpd_ec={initial_BPD_ec_conc}_t={t_eval}"

    alpha_range = np.geomspace(start=0.1, stop=1000.)  # geometric range of alpha values
    initial_target_conc_range = np.power(10, np.arange(-1, 3, dtype=float))  # range of initial [T] values

    result = kt.kprod_vs_alpha(
        config_filename=config_filename,
        protac_id=protac_id,
        t_eval=t_eval,
        alpha_range=alpha_range,
        initial_target_conc_range=initial_target_conc_range,
        initial_BPD_ec_conc=initial_BPD_ec_conc
    )
    result.to_csv(f"./saved_objects/{test_id}_kprod_vs_alpha.csv", index=False)  # save dataframe
