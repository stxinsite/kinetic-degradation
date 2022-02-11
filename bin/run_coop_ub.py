"""This script calculates target protein degradation, ternary complex formation, and
Dmax for initial concentration(s) of degrader over a range of time points.

YAML config file(s) must be saved to `./data/` folder.
"""

import numpy as np
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    """Fixed initial extracellular BPD concentration
    over a range of time points.
    """
    protac_ids = ['ACBI1']
    test_id = 'ACBI1'

    t_eval = np.linspace(0, 24)
    initial_BPD_ec_conc = 0.001  # uM

    result = kt.run_kinetic_model(
        config_files=['ACBI1_high_alpha_fast_ub_config.yml'],
        protac_IDs=protac_ids,
        t_eval=t_eval,
        initial_BPD_ec_concs=initial_BPD_ec_conc,
        return_only_final_state=False
    )
    result.to_csv(f"./saved_objects/{test_id}_BPD_ec={initial_BPD_ec_conc}_high_alpha_fast_ub_DEG.csv", index=False)

    result = kt.run_kinetic_model(
        config_files=['ACBI1_low_alpha_fast_ub_config.yml'],
        protac_IDs=protac_ids,
        t_eval=t_eval,
        initial_BPD_ec_concs=initial_BPD_ec_conc,
        return_only_final_state=False
    )
    result.to_csv(f"./saved_objects/{test_id}_BPD_ec={initial_BPD_ec_conc}_low_alpha_fast_ub_DEG.csv", index=False)

    result = kt.run_kinetic_model(
        config_files=['ACBI1_low_alpha_slow_ub_config.yml'],
        protac_IDs=protac_ids,
        t_eval=t_eval,
        initial_BPD_ec_concs=initial_BPD_ec_conc,
        return_only_final_state=False
    )
    result.to_csv(f"./saved_objects/{test_id}_BPD_ec={initial_BPD_ec_conc}_low_alpha_slow_ub_DEG.csv", index=False)

