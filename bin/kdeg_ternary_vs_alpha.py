"""
The effects on degradation of ternary complex degradation rate, and the dependence on cooperativity
at different degradation rates.
"""
import numpy as np
import pandas as pd
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_filename = 'SiTX_38406_config.yml'
    protac_id = 'ACBI1'
    t_eval = 6
    test_id = protac_id.replace(" ", "") + f't={t_eval}'

    kdeg_ternary_range = np.array([100, 500, 1000, 1500])
    alpha_range = np.geomspace(start=0.1, stop=1000)
    initial_bpd_ec_conc = 0.0005

    result = kt.kdeg_ternary_vs_alpha(
        config_filename=config_filename,
        protac_id=protac_id,
        t_eval=t_eval,
        alpha_range=alpha_range,
        kdeg_Ternary_range=kdeg_ternary_range,
        initial_BPD_ec_conc=initial_bpd_ec_conc
    )
    result.to_csv(f'./saved_objects/{test_id}_kdeg_ternary_vs_alpha_DEG.csv', index=False)