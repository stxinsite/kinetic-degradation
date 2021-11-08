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
* Ternary_Ubs: ubiquitinated ternary complexes in increasing order of length of ubiquitin chain.
  * Ternary_Ub_i: ternary complex with `i` ubiquitin molecules in chain

# Setting up a config file for a system
To model a BPD-induced, UPS-mediated target protein degradation system, you must write a config file in YAML
and save it to the `data/` folder. For example, the full path of the config file could be `data/config.yml`.

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
## Assigning array of inital values
The kinetic proofreading model supplies rate equations for the amounts of species involved in target protein degradation. The system of ODEs can be solved given initial values. For target protein degradation, the array of initial values `y0` must be an array_like object containing **in this order** the initial amounts of: `BPD_ec`, `BPD_ic`, `T`, `E3`, `BPD_T`, `BPD_E3`, `Ternary`, `Ternary_Ub_1`, ..., `Ternary_Ub_n`.

For ternary complex formation modeling, `y0` will only contain initial values for `BPD_ec`, ..., `Ternary`.

## Solving the IVP
See `bin/run_ternary_formation.py` and `src/kinetic_module/ternary_formation_test.py` for examples of how to solve the system of ODEs over time. `ternary_formation_test.py` contains a test function that takes a `params` argument. The test calls the `calc_concentrations()` function from `src/kinetic_module/kinetic_functions.py`, which wraps `scipy.integrate.solve_ivp()`

There are additional optional arguments for `calc_concentrations()` that are passed to `scipy`'s solver that can affect its performance. Set `max_step` to a small value such as 0.001 to prevent the solver from overstepping changes in species amounts. Not specifying `max_step` will run successfully, but the results may contain negative values, which is implausible as amounts must be non-negative.

To run the script from the command line:

**Linux**
```
export PYTHONPATH="$(pwd)/src/"
python bin/run_ternary_formation.py
```
**Windows**
```
set PYTHONPATH=%cd%\src\;%PYTHONPATH%
python bin/run_ternary_formation.py
```
