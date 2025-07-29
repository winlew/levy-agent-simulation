# Compare the distributions

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm
from pathlib import Path

x = np.linspace(1e-15, 0.9999, 1000)
mu_values = [1.01, 1.7, 3]
alpha_values = [0.01, 0.3, 1]

plt.figure(figsize=(10, 10))
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

blues = cm.Blues(np.linspace(0.4, 0.8, len(mu_values)))
for mu, color in zip(mu_values, blues):
    power = 1 / x**(1 / (mu - 1))
    plt.plot(x, power, color=color, label=f"Power Law (α={mu:.2f})")

reds = cm.Reds(np.linspace(0.4, 0.8, len(alpha_values)))
for alpha, color in zip(alpha_values, reds):
    exp = -np.log(x) / alpha + 1
    plt.plot(x, exp, color=color, label=f"Exponential (λ={alpha})")

gauss = norm.ppf(1 - x / 2) + 1
plt.plot(x, gauss, color='green', label="Gaussian")

plt.ylim(0.5, 9)
plt.yticks(range(1, 10))
plt.grid(linewidth=0.3)
plt.xlabel(r'$u$')
plt.ylabel("Step Length")
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'resources/distributions.pdf', format='pdf')
plt.show()