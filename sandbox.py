import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import root
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import yaml
import src.kinetic_module.kinetic_functions as kf
from src.kinetic_module.calc_full_config import KineticParameters
from src.kinetic_module.equilibrium_functions import predict_ternary

"""
Degradation vs PROTAC concentration
"""
test_id = 'PROTAC1&ACBI1'
t_eval = 6
bpd_ec = 0.001
v_ic = 5.24e-13
result_id = f"{test_id}_target=1_e3=0.1_t={t_eval}"

result = pd.read_csv(f"./saved_objects/{result_id}.csv")

result['total_bpd'] = result['total_bpd_ic'] + result['BPD_ec']

# TOTAL PROTAC
df = result[['PROTAC', 'initial_BPD_ec_conc', 'total_bpd']]
df = df.assign(initial_bpd = lambda data: np.log(data.initial_BPD_ec_conc),
               total_bpd=lambda data: np.log(data.total_bpd / v_ic))
df = df.join(pd.get_dummies(df['PROTAC']))

X = sm.add_constant(data=df[['initial_bpd', 'ACBI1']])
y = df['total_bpd']

model = sm.OLS(y, X)
linres = model.fit()
print(linres.summary())

# TERNARY COMPLEX KINETIC VS EQUILIBRIUM
result = result[['PROTAC', 'initial_BPD_ec_conc', 'total_target', 'total_bpd_ic', 'total_ternary']]
# convert amounts to concentrations
result = result.assign(total_ternary=lambda df: df.total_ternary / v_ic,
                       total_bpd_ic=lambda df: df.total_bpd_ic / v_ic,
                       total_target=lambda df: df.total_target / v_ic)
result['ternary_equilibrium'] = 100.0

for idx, dat in result.iterrows():
    total_target = dat['total_target']
    total_protac = dat['total_bpd_ic']
    total_e3 = 0.1
    if dat['PROTAC'] == 'PROTAC 1':
        kd_target = 8.54
        kd_e3 = 1.23e-2
        alpha = 3.2
    elif dat['PROTAC'] == 'ACBI1':
        kd_target = 9.26
        kd_e3 = 0.0694
        alpha = 26
    else:
        raise ValueError('PROTAC not available.')

    ternary = predict_ternary(total_target, total_protac, total_e3, kd_target, kd_e3, alpha)
    result.at[idx, 'ternary_equilibrium'] = ternary

n = 10

fake_df = pd.DataFrame({
    'BPD_ec': np.random.random(n),
    'BPD_ic': np.random.random(n),
    'T': np.random.random(n),
    'E3': np.random.random(n),
    'BPD_T': np.random.random(n),
    'BPD_E3': np.random.random(n),
    'Ternary': np.random.random(n),
    'T_Ub_1': np.random.random(n),
    'BPD_T_Ub_1': np.random.random(n),
    'Ternary_Ub_1': np.random.random(n),
    'T_Ub_2': np.random.random(n),
    'BPD_T_Ub_2': np.random.random(n),
    'Ternary_Ub_2': np.random.random(n)
})







protac_id = 'ACBI1'
t = 1
bpd_ec = 0.001
result_id = f"{protac_id}_bpd_ec={bpd_ec}_t={t}_kub_vs_alpha_DEG"

result1 = pd.read_csv("./saved_objects/ACBI1_bpd_ec=0.001_t=1_kub_vs_alpha_DEG.csv")
result2 = pd.read_csv("./saved_objects/ACBI1_bpd_ec=0.001_t=6_kub_vs_alpha_DEG.csv")
result = pd.concat([result1, result2])
alpha_range = result['alpha'].unique()

result = result[['alpha', 'kub', 't', 'BPD_ic']]
result['BPD_ic_conc'] = result['BPD_ic'] / 5.24e-13
result = result.melt(id_vars=['alpha', 'kub', 't'], value_vars='BPD_ic_conc')

sns.set_style("whitegrid")
fig, ax = plt.subplots()
p = sns.lineplot(
    data=result,
    x='alpha',
    y='value',
    hue='kub',
    style='t',
    ci=None,
    palette='Set2',
    linewidth=2,
    alpha=1,
    ax=ax
)
p.tick_params(labelsize=12)
plt.xscale('log')
plt.xlim(alpha_range.min(), alpha_range.max())
plt.xlabel(r'Cooperativity $\alpha$')
plt.ylabel(f'Free intracellular PROTAC ($\mu$M)')
handles, labels = ax.get_legend_handles_labels()
for i in range(1, 5):
    labels[i] = r"$k_{ub} = $" + fr"${labels[i]}/h$"
labels[5] = ''
labels[6] = f"t = {labels[6]}h"
labels[7] = f"t = {labels[7]}h"
ax.legend(handles=handles[1:], labels=labels[1:], loc='lower left', borderaxespad=0.25)
plt.setp(ax.get_legend().get_texts(), fontsize='8')  # for legend text


with open(f"./data/SiTX_38406_config.yml") as file:
    acbi1_config = yaml.safe_load(file)  # load original config

acbi1_params = KineticParameters(acbi1_config)
print(acbi1_params)

with open(f"./data/SiTX_38404_config.yml") as file:
    protac1_config = yaml.safe_load(file)  # load original config

protac1_params = KineticParameters(protac1_config)
print(protac1_params)

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
