# Kinetic Modeling of Target Protein Degradation
This repo implements a kinetic proofreading model of protein degradation via the ubiquitin-proteasome system (UPS) as developed by Bartlett et al. (2013) [in this paper](https://doi.org/10.1007/s10928-020-09722-z). The model calculates the amounts of species involved a system over time by solving a system of ordinary differential equations describing the rates of change in species amounts. Scripts in this repo can:
- Solve for species amounts over time given initial values
- Calculate target protein degradation (TPD) and ternary complex formation (TCF) relative to baseline target protein amount
- Model and visualize TPD and TCF with a range of parameter values

## Prerequisites
Dependencies can be found in `requirements.txt` can be installed with
```sh
pip install -r requirements.txt
```

## Notation
We maintain a similar notation as used by Bartlett et al. and denote the amounts of species involved in the UPS as follows. For consistency, ensure that all initial values and parameter measurements **are in the same units**. We will use **umol** for amount units:

* BPD_ec: unbound extracellular Bispecific Protein Degrader.
* BPD_ic: unbound intracellular Bispecific Protein Degrader.
* T: unbound Target protein.
* E3: unbound E3 ligase.
* BPD_T: BPD-T binary complex.
* BPD_E3: BPD-E3 binary complex.
* Ternary: T-BPD-E3 ternary complex.
* Ternary_Ub_i: ternary complex with *i* ubiquitin molecules in poly-ubiquitin chain.

# Run Kinetic Model
## Provide a config file for a system
To model a target protein + degrader + E3 ligase system, you must write a config file in YAML or JSON and save it to the `data/` folder.

The config file must contain the following keys (with units in parentheses):
<details>
  <summary>Click to expand</summary>

  <blockquote>

  <details>
    <summary>Kinetic rate parameters</summary>

    ```yaml
    - alpha: ternary complex cooperativity
    - Kd_T_binary (uM): equilibrium dissociation constant of BPD-T binary complex
    - kon_T_binary (L/umol/h): kon of BPD + T -> BPD-T
    - koff_T_binary (1/h): koff of BPD-T -> BPD + T
    - Kd_T_ternary (uM): equilibrium dissociation constant of T in ternary complex
    - kon_T_ternary (L/umol/h): kon of BPD-E3 + T -> T-BPD-E3
    - koff_T_ternary (1/h): koff of T-BPD-E3 -> BPD-E3 + T
    - Kd_E3_binary (uM): equilibrium dissociation constant of BPD-E3 binary complex
    - kon_E3_binary (L/umol/h): kon of BPD + E3 -> BPD-E3
    - koff_E3_binary (1/h): koff of BPD-E3 -> BPD + E3
    - Kd_E3_ternary (uM): equilibrium dissociation constant of E3 in ternary complex
    - kon_E3_ternary (L/umol/h): kon of BPD-T + E3 -> T-BPD-E3
    - koff_E3_ternary (1/h): koff of T-BPD-E3 -> BPD-T + E3
    ```
  </details>

  <details>
    <summary>Other parameters</summary>

    ```yaml
    - n: number of ubiquitination steps before degradation
    - MTT_deg (h): mean transit time of degradation
    - ktransit_UPS (1/h): transit rate for delay between each ubiquitination step ((n+1) / MTT_deg)
    - fu_ec: fraction unbound extracellular BPD
    - fu_ic: fraction unbound intracellular BPD
    - PS_cell (L/h): permeability-surface area product
    - kprod_T (umol/h): baseline target protein production rate (Conc_T_base * Vic * kdeg_T)
    - kdeg_T (1/h): baseline target protein degradation rate
    - Conc_T_base (uM): baseline target protein concentration
    - Conc_E3_base (uM): baseline E3 concentration
    - num_cells: number of cells in system
    - Vic (L): intracellular volume
    - Vec (L): extracellular volume
    ```
  </details>

  </blockquote>
</details>

A sufficient number of kinetic rate parameters **must** be defined. The `KineticParameters` class can solve for the rest of the unknown dependent parameters given sufficient information using the ratios `Kd = koff / kon` and `alpha = Kd_binary / Kd_ternary` and test for consistency with these ratios if all parameters are provided.

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
See `bin/run_ternary_formation.py` and `src/kinetic_module/kinetic_tests.py` for examples of how to solve the system of ODEs over time. `kinetic_tests.py` contains a test function `solve_ternary_formation()` that takes a `params` argument. The test calls the `calc_concentrations()` function from `src/kinetic_module/kinetic_functions.py`, which wraps `scipy.integrate.solve_ivp()`.

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
