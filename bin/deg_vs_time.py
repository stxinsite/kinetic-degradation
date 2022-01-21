"""This script calculates target protein degradation, ternary complex formation, and
Dmax for initial concentration(s) of degrader over a range of time points.

YAML config file(s) must be saved to `./data/` folder.
"""

import numpy as np
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_files = ['SiTX_38404_config.yml', 'SiTX_38406_config.yml']
    protac_ids = ['PROTAC 1', 'ACBI1']
    test_id = 'PROTAC1&ACBI1'

    """Fixed initial intracellular BPD concentration
    over a range of time points.
    """
    t_eval = np.linspace(0, 1, num=100)
    initial_BPD_conc = 0.001

    result = kt.run_kinetic_model(
        config_files=config_files,
        protac_IDs=protac_ids,
        t_eval=t_eval,
        initial_BPD_ec_concs=initial_BPD_conc,
        return_only_final_state=False
    )
    result.to_csv(f"./saved_objects/{test_id}_bpd_ec={initial_BPD_conc}_t={np.max(t_eval)}_DEG.csv", index=False)
