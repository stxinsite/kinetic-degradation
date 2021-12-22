# Introduction

This repo implements a kinetic proofreading model of protein degradation via the ubiquitin-proteasome system (UPS) motivated by the work of Bartlett et al. (2020)<sup>[1](#bartlett)</sup>. Given an initial state of species amounts, 
this repo can perform the following functions:
- solve a system of ODEs for the system state over time
- calculate target protein degradation, ternary complex formation, Dmax (percent maximum degradation)
- compute species totals over time

For mathematical details, see technical document [here](https://www.overleaf.com/read/zmcpqnbknhqs).

## Prerequisites

Dependencies can be found in `environment.yml`. To create an environment satisfying all the dependencies with [Anaconda](https://docs.anaconda.com/anaconda/install/index.html), make sure that the current working directory contains
`environment.yml`, then run the following command:
```commandline
conda env create
```

Activate the created environment with:
```commandline
conda activate kinetic-degradation
```

## Notation

We maintain a similar notation as used by Bartlett et al. and denote the amounts of species involved in the UPS as follows. Ensure that all initial values and model parameters have **_consistent units_**. We will use micromolar for concentration, liter for volume, and hour for time.

<details>
  <summary>Model species notation</summary>

  * BPD_ec: extracellular Bispecific Protein Degrader.
  * BPD_ic: intracellular Bispecific Protein Degrader.
  * T: unbound Target protein.
  * E3: unbound E3 ligase.
  * BPD_T: BPD-T binary complex.
  * BPD_E3: BPD-E3 binary complex.
  * Ternary: T-BPD-E3 ternary complex.
  * T_Ub_i: Target protein with *i* ubiquitin molecules attached.
  * BPD_T_Ub_i: BPD-T binary complex with *i* ubiquitin molecules attached.
  * Ternary_Ub_i: ternary complex with *i* ubiquitin molecules attached.
</details>

# Applying the model to a system

## Providing a system config file

To model a target protein degradation system, you must provide a config file in a format such as YAML that can be read into a Python dictionary data structure and save it to the `./data/` folder.

The config file must contain the following keys (with units in parentheses):
<details>
  <summary>Kinetic rate constants and Binding affinity</summary>

  ```yaml
  - alpha: ternary complex cooperativity
  - Kd_T_binary (uM): equilibrium dissociation constant of BPD-T binary complex
  - kon_T_binary (1/uM/h): kon of BPD + T -> BPD-T
  - koff_T_binary (1/h): koff of BPD-T -> BPD + T
  - Kd_T_ternary (uM): equilibrium dissociation constant of T in ternary complex
  - kon_T_ternary (1/uM/h): kon of BPD-E3 + T -> T-BPD-E3
  - koff_T_ternary (1/h): koff of T-BPD-E3 -> BPD-E3 + T
  - Kd_E3_binary (uM): equilibrium dissociation constant of BPD-E3 binary complex
  - kon_E3_binary (1/uM/h): kon of BPD + E3 -> BPD-E3
  - koff_E3_binary (1/h): koff of BPD-E3 -> BPD + E3
  - Kd_E3_ternary (uM): equilibrium dissociation constant of E3 in ternary complex
  - kon_E3_ternary (1/uM/h): kon of BPD-T + E3 -> T-BPD-E3
  - koff_E3_ternary (1/h): koff of T-BPD-E3 -> BPD-T + E3
  ```
</details>

<details>
  <summary>Other parameters</summary>

  ```yaml
  - n: number of ubiquitination steps before proteasomal degradation
  - kub (1/h): rate of ubiquitination
  - kde_ub (1/h): rate of de-ubiquitination
  - kdeg_UPS (1/h): rate of proteasomal degradation for poly-ubiquitinated T and BPD-T
  - kdeg_Ternary (1/h): rate of proteasomal degradation for poly-ubiquitinated Ternary
  - fu_ec: fraction unbound extracellular BPD
  - fu_ic: fraction unbound intracellular BPD
  - PS_cell (L/h): permeability-surface area product
  - kprod_T (umol/h): [Optional] intrinsic target protein production rate. Will be calculated and set to Conc_T_base * Vic * kdeg_T
  - kdeg_T (1/h): intrinsic target protein degradation rate
  - Conc_T_base (uM): baseline target protein concentration
  - Conc_E3_base (uM): baseline E3 concentration
  - num_cells: number of cells in system
  - Vic (L): intracellular volume
  - Vec (L): extracellular volume
  ```
</details>

A sufficient number of kinetic rate parameters must be defined and those that are provided must be consistent. That is, a config is sufficient and consistent if all kinetic rate constants can either be derived from or satisfy the following ratios:
```
Kd = koff / kon
alpha = Kd_binary / Kd_ternary
```
## Special cases of target protein degradation

### Ternary complex formation

Ternary complex formation is just a special case of the kinetic proofreading model wherein target protein ubiquitination and proteasomal degradation is prevented. This is equivalent to setting the following parameters to 0 in the config file:
```
- n
- kub
- kde_ub
- kdeg_UPS
- kdeg_Ternary
```

### Intracellular systems

It is possible to model entirely intracellular systems by setting the following parameters to 0 in the config file:
```
- PS_cell
```

# The Kinetic Proofreading Model as an Initial Value Problem

This repo implements a mechanistic mathematical model as a system of ordinary differential equations that describe the change in amounts of species involved in a UPS over time given an initial condition.

## Simulating target protein degradation by solving an IVP

See files in `bin/` for examples of how to simulate a UPS. 

- `run_kinetic_model.py`: simulates a UPS for
  - fixed initial concentration of degrader over time
  - fixed time point over initial concentrations of degrader
- `Kd_vs_alpha.py`: analyzes effect of cooperativity across binary target `Kd` on degradation for fixed initial concentration of degrader and time point
- `ktransit_vs_alpha.py`: analyzes effect of cooperativity across ubiquitination rate `kub` on degradation for fixed initial concentration of degrader and time point

# Tutorial  

To run scripts from the command line, first run:

**Linux**
```commandline
export PYTHONPATH="$(pwd)/src/"
```
**Windows**
```commandline
set PYTHONPATH=%cd%\src\;%PYTHONPATH%
```

In your current working directory, create a folder named `saved_objects/`; this is where results will be stored.

## Example 
```commandline
python bin/run_kinetic_model.py
```


#### References
<a name="bartlett">1</a>: https://doi.org/10.1007/s10928-020-09722-z