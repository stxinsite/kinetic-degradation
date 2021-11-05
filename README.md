# kinetic-degradation
This repo implements a kinetic proofreading model of protein degradation via the ubiquitin-proteasome system (UPS) as
developed by Bartlett et al. (2013) [in this paper](https://doi.org/10.1007/s10928-020-09722-z).
We maintain a similar notation and denote the species involved in ternary complex formation and target protein degradation as follows:

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
<details>
  <summary>Click to expand</summary>

  - alpha: ternary complex cooperativity
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
</details>

## Ternary complex formation as a special case
For modeling protein degradation, all the parameters in the previous section must be specified. If the process of interest is ternary complex formation, this is just a special case of the kinetic proofreading model in which no ubiquitination or degradation occurs in the cell. To model ternary complex formation, set the following parameters to 0 in the config file:
- n
- MTT_deg
- ktransit_UPS
- kprod_T
- kdeg_T

## Intracellular special case
If extracellular BPD is not of interest (i.e., the BPD has been introduced into cells), set the following parameters to 0 in the config file:
- PS_cell
- num_cells

Although the extracellular environment is not of interest, the `Vec` parameter still must be positive to avoid division by zero in the equation for the `BPD_ec` rate. Setting the above parameters to 0 ensures that the rate is always 0, and thus the amount of `BPD_ec` remains constant over time.

The initial value for `BPD_ic` will then presumably be greater than zero.

# Modeling kinetic proofreading as solving an initial value problem
`max_step` for IVP solver. Set to small value ~ 0.001, because otherwise the solver will overshoot and results at some time points may be negative, which is implausible.
 `calc_concentrations()`
