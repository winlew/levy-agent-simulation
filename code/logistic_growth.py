# Visualize logistic growth function and make phase plane plot

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

growth_rate = 1.0
carrying_capacity = 100.0
x0 = 0.5    
t = np.linspace(0, 12, 200)

def logistic_growth(t, r, K, x0):
    return K / (1 + ((K - x0) / x0) * np.exp(-r * t))

x_t = logistic_growth(t, growth_rate, carrying_capacity, x0)

x_vals = np.linspace(0, carrying_capacity*1.1, 200)
dxdt = growth_rate * x_vals * (1 - x_vals / carrying_capacity)

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Logistic growth plot
axs[0].plot(t, x_t)
axs[0].set_xlabel("Time")
axs[0].set_ylabel(r'$N$')
axs[0].grid(True, linewidth=0.3)
axs[0].axhline(carrying_capacity, ls='--', linewidth=1, color="#3B3B3B", label=r'$K$')
axs[0].legend()

# Phase plane plot
axs[1].plot(x_vals, dxdt)
axs[1].set_xlabel(r'$N$')
axs[1].set_ylabel(r'$\dot{N}$')
axs[1].grid(True, linewidth=0.3)
axs[1].scatter(carrying_capacity, 0, color="#646464", edgecolor='black', s=45, label='Fixed point', zorder=5)

axs[1].legend(handletextpad=0.2)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'resources/logistic_growth.pdf', format='pdf')