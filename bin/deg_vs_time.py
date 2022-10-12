"""This script calculates target protein degradation, ternary complex formation, and
Dmax for initial concentration(s) of degrader over a range of time points.

YAML config file(s) must be saved to `./data/` folder.
"""

import numpy as np
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_files = [
        'ACBI1_WT_config.yml',
        'ACBI1_E1420S_config.yml',
        'ACBI1_L1465S_config.yml',
        'ACBI1_R69Q_config.yml',
        'ACBI1_Y112F_config.yml',
        # 'SiTX_38404_config.yml', 
        # 'SiTX_38404_star_config.yml',
        # 'SiTX_38406_config.yml',
        # 'SiTX_38404_kdeg_ternary_config.yml',
        # 'SiTX_38404_star_kdeg_ternary_config.yml',
        # 'SiTX_38406_kdeg_ternary_config.yml',
    ]
    protac_ids = [
        'WT',
        'E1420S',
        'L1465S',
        'R69Q',
        'Y112F',
        # 'PROTAC 1', 
        # 'PROTAC 1*',
        # 'ACBI1',
        # 'PROTAC 1',
        # 'PROTAC 1*',
        # 'ACBI1',
    ]

    test_id = 'deg_vs_time_nanobret'

    """Fixed initial intracellular BPD concentration
    over a range of time points.
    """
    # t_eval = np.linspace(0, 6, num=50)
    t_eval = [0, 6, 12]
    initial_BPD_conc = 0.001

    result = kt.run_kinetic_model(
        config_files=config_files,
        protac_IDs=protac_ids,
        t_eval=t_eval,
        initial_BPD_ec_concs=initial_BPD_conc,
        return_only_final_state=False
    )

    result.to_csv(f"./saved_objects/{test_id}_bpd_ec={initial_BPD_conc}_t={int(np.max(t_eval))}.csv", index=False)
