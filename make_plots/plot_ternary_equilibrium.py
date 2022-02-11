import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from kinetic_module.equilibrium_functions import predict_ternary

plt.rcParams["axes.labelsize"] = 14
plt.rcParams["figure.figsize"] = (4, 3.5)
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.direction"] = 'in'
plt.rcParams["ytick.labelsize"] = 12

n = 50

total_target = 1.0
total_protac = np.logspace(base=10.0, start=-4, stop=2, num=n)
total_e3 = 0.1
kd_target = 9.26
kd_e3 = 0.0694
alpha = 26
ternary = np.empty(n)

for idx, protac in enumerate(total_protac):
    ternary[idx] = predict_ternary(
        total_target, protac, total_e3, kd_target, kd_e3, alpha
    )

df = pd.DataFrame({
    'total_protac': total_protac,
    'ternary': ternary
})

sns.set_style("white")

fix, ax = plt.subplots()

sns.lineplot(
    data=df,
    x='total_protac',
    y='ternary',
    linewidth=2,
    ax=ax
)

plt.xscale('log')
plt.xlim(total_protac.min(), total_protac.max())
plt.ylim(bottom=0)
plt.xlabel(r'ACBI1 ($\mu$M)')
plt.ylabel(r'Ternary complex ($\mu$M)')

plt.savefig("plots/ACBI1_ternary_equilibrium.png", bbox_inches='tight', dpi=1200)
