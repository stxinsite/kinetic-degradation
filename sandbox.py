import numpy as np
from scipy.optimize import root
import yaml
import src.kinetic_module.kinetic_functions as kf
from src.kinetic_module.calc_full_config import KineticParameters

with open(f"./data/SiTX_38406_config.yml") as file:
    params = yaml.safe_load(file)  # load original config

Params = KineticParameters(params)
Params.is_fully_defined()
params = Params.get_dict()

t = np.linspace(0, 6)

BPD_ec = 0.1 * params['Vec']
BPD_ic = 0
T = params['Conc_T_base'] * params['Vic']
E3 = params['Conc_E3_base'] * params['Vic']
BPD_T = 0
BPD_E3 = 0
Ternary = 0
Ternary_Ubs = [0] * params['n']
y0 = np.array([BPD_ec, BPD_ic, T, E3, BPD_T, BPD_E3, Ternary] + Ternary_Ubs)

result = kf.calc_concentrations(params, t, y0, max_step = 0.001)

init_guess = result.y[:,-1]  # system state at the last time point
steady_state = kf.solve_steady_state(init_guess, params)
steady_state

T_total_steady_state = np.sum(np.concatenate((steady_state[[2,4]], steady_state[6:])))
T_total_baseline = np.sum(np.concatenate((y0[[2,4]], y0[6:])))
Dmax_manual = 1 - T_total_steady_state / T_total_baseline
Dmax = kf.calc_Dmax(params, t, y0)

Dmax_manual == Dmax
Dmax

result2 = kf.calc_concentrations(params, t, y0, max_step = 0.001,
                                 T_total_baseline = T_total_baseline, T_total_steady_state = T_total_steady_state
                                 )

result2.t_events[0][0]
yhalf = result2.y_events[0][0]
T_total_half = np.sum(np.concatenate((yhalf[[2,4]], yhalf[6:])))
C = 0.5 * (T_total_baseline - T_total_steady_state)

T_total_half == C
T_total_half


t_half = kf.calc_t_half(params, t, y0)  # 0.783
t_half

def f(t, y0, params, T_total_half):
    result = kf.calc_concentrations(params, [t], y0, max_step = 0.001)
    T_total_at_t = np.sum(np.concatenate((result.y[[2,4]], result.y[6:])))
    return T_total_at_t - T_total_half

def fprime(t, y0, params, T_total_half):
    result = kf.calc_concentrations(params, [t], y0, max_step = 0.001)

def wrap_kinetic_rates(arr, params):
    return kf.kinetic_rates(params, *arr)

def wrap_jac_kinetic_rates(arr, params):
    return kf.jac_kinetic_rates(params, *arr)

roots = root(wrap_kinetic_rates, init_guess, jac = wrap_jac_kinetic_rates, args = (params,))
if (not roots.success) or (np.any(roots.x < 0)):
    roots = root(wrap_kinetic_rates, init_guess, jac = wrap_jac_kinetic_rates, args = (params,), method = 'lm')

init_guess
roots.success
roots.message
roots.x
roots.fun

Dmax

 0.5 * (T_total_baseline - T_total_steady_state) / T_total_baseline
