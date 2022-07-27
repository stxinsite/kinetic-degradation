"""This script calculates target protein degradation, ternary complex formation, and
Dmax across initial concentrations of degrader.

YAML config file(s) must be saved to `./data/` folder.
"""

import numpy as np
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_files = [
        'SiTX_38404_kdeg_ternary_low_E3_config.yml',
        'SiTX_38406_kdeg_ternary_low_E3_config.yml'
    ]
    protac_ids = [
        'PROTAC 1', 
        'ACBI1'
    ]

    test_id = 'deg_ternary_protac_vs_protac_high_kdeg_ternary'

    """Vary initial extracellular BPD concentrations and
    solve to a fixed time point.
    """
    t_eval = 6  # hours
    initial_BPD_ec_concs = np.logspace(base=10.0, start=-4, stop=2)  # uM

    result = kt.run_kinetic_model(
        config_files=config_files,
        protac_IDs=protac_ids,
        t_eval=t_eval,
        initial_BPD_ec_concs=initial_BPD_ec_concs,
        return_only_final_state=True
    )
    
    result.to_csv(f"./saved_objects/{test_id}_t={t_eval}.csv", index=False)
