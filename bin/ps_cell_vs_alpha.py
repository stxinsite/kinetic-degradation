"""
The effects on degradation of permeability surface area and cooperativity.
"""

import numpy as np
import pandas as pd
import kinetic_module.kinetic_tests as kt

if __name__ == '__main__':
    config_filename = 'SiTX_38406_config.yml'
    protac_id = 'ACBI1'
    t_eval = 6
    test_id = protac_id.replace(" ", "") + f't={t_eval}'

    ps_cell_range = np.array([2.5e-12, 2.5e-11, 2.5e-10, 2.5e-9])
    alpha_range = np.geomspace(start=0.1, stop=1000)
    initial_bpd_ec_conc = 0.0005

    result = kt.ps_cell_vs_alpha(
        config_filename=config_filename,
        protac_id=protac_id,
        t_eval=t_eval,
        alpha_range=alpha_range,
        ps_cell_range=ps_cell_range,
        initial_BPD_ec_conc=initial_bpd_ec_conc
    )
    result.to_csv(f'./saved_objects/{test_id}_ps_cell_vs_alpha_DEG.csv', index=False)