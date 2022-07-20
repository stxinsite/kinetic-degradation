"""
The effects on degradation of ternary complex degradation rate, and the dependence on cooperativity
at different degradation rates.
"""
import numpy as np
import pandas as pd
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_filename = 'ACBI1_kdeg_ternary_config.yml'
    protac_id = 'ACBI1'
    t_eval = 6
    kd = 0.01
    test_id = f'{protac_id}_t={t_eval}_kd={kd}'

    kdeg_target = 60
    kdeg_ternary_range = [kdeg_target / (2 ** i) for i in range(11)]
    kdeg_ternary_range.append(0)
    alpha_range = np.geomspace(start=0.1, stop=1000, num=150)
    initial_bpd_ec_conc = 0.001

    # result will be a DataFrame with 11 * 150 rows
    result: pd.DataFrame = kt.kdeg_ternary_vs_alpha(
        config_filename=config_filename,
        protac_id=protac_id,
        t_eval=t_eval,
        alpha_range=alpha_range,
        kdeg_Ternary_range=kdeg_ternary_range,
        initial_BPD_ec_conc=initial_bpd_ec_conc
    )

    result.to_csv(
        f'./saved_objects/{test_id}_kdeg_ternary_vs_alpha.csv', index=False)
