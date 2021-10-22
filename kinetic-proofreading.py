import scipy.integrate as integrate

kon_T_binary = 3600.
koff_T_binary = 327.6
Kd_T_binary =
kon_T_ternary
koff_T_ternary
Kd_T_ternary
kon_E3_binary
koff_E3_binary
Kd_E3_binary
kon_E3_ternary
koff_E3_ternary
Kd_E3_ternary
alpha
ktransit_UPS
n
MTT_deg
fu_c
fu_ec
fu_ic
F
ka
CL
Vc
Q
Vp
PS_cell
PSV_tissue
MW_BPD

BPD_ev
BPD_c
BPD_p
BPD_ec
BPD_ic
T
E3
BPD_T
BPD_E3
Ternary
Ternary_Ub_1, Ternary_Ub_2, Ternary_Ub_3, Ternary_Ub_4

kprod_T
kdeg_T
Conc_T_base
Conc_E3_base
num_cells
Vic
Vec
BW

dBPDicdt = PS_cell * (fu_ec * BPD_ec / Vec - fu_ic * BPD_ic / Vic) - \  # permeability times difference in free extracellular BPD and free intracellular BPD; free BPD leaving/entering cell
           kon_T_binary * fu_ic * BPD_ic * T / Vic + \  # BPD binds with Target in binary
           koff_T_binary * BPD_T - \  # BPD unbinds from Target in binary
           kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \  # BPD binds with E3 in binary
           koff_E3_binary * BPD_E3 + \  # BPD unbinds from E3 in binary
           kdeg_T * BPD_T  # BPD unbinds from degraded Target

dTargetdt = kprod_T - \  # Target protein production rate at baseline
            kdeg_T * T - \  # Target protein gets degraded
            kon_T_binary * fu_ic * BPD_ic * T / Vic + \  # Target binds with BPD in binary
            koff_T_binary * BPD_T - \  # Target unbinds from BPD in binary
            kon_T_ternary * BPD_E3 * T / Vic + \  # Target binds with BPD-E3 in ternary
            koff_T_ternary * (Ternary + Ternary_Ub_1 + Ternary_Ub_2 + Ternary_Ub_3 + Ternary_Ub_4)  # Target unbinds from Ternary

dE3dt = -kon_E3_binary * fu_ic * BPD_ic * E3 / Vic + \  # E3 binds with BPD in binary
        koff_E3_binary * BPD_E3 - \  # E3 unbinds from BPD in binary
        kon_E3_ternary * BPD_T * E3 / Vic + \  # E3 binds with BPD-Target in ternary
        koff_E3_ternary * (Ternary + Ternary_Ub_1 + Ternary_Ub_2 + Ternary_Ub_3 + Ternary_Ub_4)  # E3 unbinds from Ternary
