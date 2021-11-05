# kinetic-degradation
This repo implements a kinetic proofreading model of protein degradation via the ubiquitin-proteasome system (UPS).
We denote the species involved in ternary complex formation and target protein degradation as follows:

* BPD_ec: unbound extracellular Bispecific Protein Degrader.
* BPD_ic: unbound intracellular Bispecific Protein Degrader.
* T: unbound Target protein.
* E3: unbound E3 ligase.
* BPD_T: BPD-T binary complex.
* BPD_E3: BPD-E3 binary complex.
* Ternary: T-BPD-E3 ternary complex.
* Ternary_Ubs: ubiquitinated ternary complex in increasing order of length of ubiquitin chain.
  * Ternary_Ub_i: ternary complex with `i` ubiquitin molecules in chain

# Setting up a config file for a system
To model a BPD-induced, UPS-mediated target protein degradation system, you must write a config file in YAML
and save it to the `model_configs\` folder. For example, the full path of the config file could be `model_configs\config.yml`.

The config file must contain the following keys:
- alpha: cooperativity
- Kd_T_binary: equilibrium dissociation constant of BPD-T binary complex
- kon_T_binary: kon of BPD + T -> BPD-T
- koff_T_binary: koff of BPD-T -> BPD + T
- Kd_T_ternary: equilibrium dissociation constant of T in ternary complex
- kon_T_ternary: kon of BPD-E3 + T -> T-BPD-E3
- koff_T_ternary: koff of T-BPD-E3 -> BPD-E3 + T
- Kd_E3_binary: equilibrium dissociation constant of BPD-E3 binary complex
- kon_E3_binary: kon of BPD + E3 -> BPD-E3
- koff_E3_binary: koff of BPD-E3 -> BPD + E3
- Kd_E3_ternary: equilibrium dissociation constant of E3 in ternary complex
- kon_E3_ternary: kon of BPD-T + E3 -> T-BPD-E3
- koff_E3_ternary: koff of T-BPD-E3 -> BPD-T + E3
- n: number of ubiquitination steps before degradation
- MTT_deg: mean transit time of degradation
- ktransit_UPS: transit rate for delay between each ubiquitination step
- fu_ec: fraction unbound extracellular BPD
- fu_ic: fraction unbound intracellular BPD
- PS_cell: permeability-surface area product
- kprod_T: baseline target protein production rate
- kdeg_T: baseline target protein degradation rate
- Conc_T_base: baseline target protein concentration
- Conc_E3_base: baseline E3 concentration
- num_cells: number of cells in system
- Vic: intracellular volume
- Vec: extracellular volume
