# Visualize logistic growth function and make phase plane plot

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import LATEX_TEXTWIDTH

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 10,
    "font.family": "serif"
})
fig, axs = plt.subplots(1, 2, figsize=(LATEX_TEXTWIDTH, LATEX_TEXTWIDTH/2))

growth_rate = 1.0
carrying_capacity = 100.0
x0 = 0.5    
t = np.linspace(0, 12, 200)

def logistic_growth(t, r, K, x0):
    return K / (1 + ((K - x0) / x0) * np.exp(-r * t))

x_t = logistic_growth(t, growth_rate, carrying_capacity, x0)

x_vals = np.linspace(0, carrying_capacity*1.1, 200)
dxdt = growth_rate * x_vals * (1 - x_vals / carrying_capacity)

# Logistic growth plot
axs[0].plot(t, x_t)
axs[0].set_xlabel("Time")
axs[0].set_ylabel('$N$', labelpad=2)
axs[0].grid(True, ls="--", linewidth=0.4)
axs[0].axhline(carrying_capacity, ls='--', linewidth=2, color="#E98542", label='$K$')
legend = axs[0].legend(fancybox=True, edgecolor='gray', borderpad=0.3, facecolor='white', frameon=True, framealpha=1)
legend.get_frame().set_linewidth(0.5)

# Phase plane plot
axs[1].plot(x_vals, dxdt)
axs[1].set_xlabel('$N$')
axs[1].set_ylabel('$\dot{N}$', labelpad=2)
axs[1].grid(True, ls="--", linewidth=0.4)
axs[1].scatter(carrying_capacity, 0, color="#000000", edgecolor='black', linewidth=0.8, s=20, label='Stable Fixed Point', zorder=5)
axs[1].scatter(0, 0, color="#FFFFFF00", edgecolor='black', linewidth=0.8, s=20, label='Unstable Fixed Point', zorder=5)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'resources/logistic_growth.pdf', format='pdf')