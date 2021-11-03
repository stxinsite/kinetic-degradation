import numpy as np

"""GLOBAL VARIABLES"""
# RATE AND BINDING PARAMETERS
alpha = 1.

Kd_T_binary = 0.091
kon_T_binary = 3600.
koff_T_binary = kon_T_binary * Kd_T_binary

Kd_T_ternary = Kd_T_binary / alpha
koff_T_ternary = 327.6
kon_T_ternary = koff_T_ternary / Kd_T_ternary

Kd_E3_binary = 3.1
kon_E3_binary = 3600.
koff_E3_binary = kon_E3_binary * Kd_E3_binary

Kd_E3_ternary = Kd_E3_binary / alpha
koff_E3_ternary = 11160.
kon_E3_ternary = koff_E3_ternary / Kd_E3_ternary

# DEGRADATION PARAMETERS
n = 3  # set to 0 for ternary formation
MTT_deg = 0.0015
ktransit_UPS = (n + 1) / MTT_deg  # set to 0 for ternary formation
fu_c = np.nan
fu_ec = 1.
fu_ic = 1.
F = np.nan
ka = np.nan
CL = np.nan
Vc = np.nan
Q = np.nan
Vp = np.nan
PS_cell = 1e-12  # set to 0 for ternary formation
PSV_tissue = np.nan
MW_BPD = 947.

# PHYSIOLOGICAL SYSTEM PARAMETERS
kdeg_T = 0.058  # set to 0 for ternary formation
Conc_T_base = 0.001
Conc_E3_base = 0.1
num_cells = 5e3
Vic = 5e-13
Vec = 2e-4
kprod_T = Conc_T_base * Vic * kdeg_T  # kprod_T = T * kdeg_T
BW = np.nan

"""INITIAL VALUES"""
BPD_ev = 0
BPD_c = 0
BPD_p = 0
BPD_ec = 0  # nM * Vec / 1000
BPD_ic = 0  # nM * Vic / 1000
T = Conc_T_base * Vic
E3 = Conc_E3_base * Vic
BPD_T = 0
BPD_E3 = 0
Ternary = 0
Ternary_Ubs = [0] * n  # where i = 0 is un-ubiquitinated Ternary
