# Creates plots to compare the agent step length distributions

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.cm as cm
from pathlib import Path
from config import LATEX_TEXTWIDTH

mu_values = [1.01, 2, 3]
alpha_values = [0.01, 0.3, 1]

plt.figure(figsize=(LATEX_TEXTWIDTH, LATEX_TEXTWIDTH))
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 10,
    "font.family": "serif"
})
blues = cm.Blues(np.linspace(0.4, 0.8, len(mu_values)))
reds = cm.Reds(np.linspace(0.4, 0.8, len(alpha_values)))


# 1. inverse transform sampling: step length vs random number from 0 to 1 plot

x = np.linspace(1e-15, 0.9999, 1000)

for mu, color in zip(mu_values, blues):
    power = 1 / x**(1 / (mu - 1))
    plt.plot(x, power, color=color, label=f"Power Law ($\mu$={mu:.2f})")

for alpha, color in zip(alpha_values, reds):
    exp = -np.log(x) / alpha + 1
    plt.plot(x, exp, color=color, label=f"Exponential ($\\alpha$={alpha})")

gauss = norm.ppf(1 - x / 2) + 1
plt.plot(x, gauss, color='green', label="Gaussian")

plt.ylim(0.5, 9)
plt.yticks(range(1, 10))
plt.grid(ls="--", linewidth=0.4)
plt.xlabel('$u$')
plt.ylabel("Step Length")
legend = plt.legend(loc='upper center', fancybox=True, edgecolor='gray', borderpad=0.3, facecolor='white', frameon=True, framealpha=1)
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'resources/distributions.pdf', format='pdf')


# 2. normal probability of step length plot

limit = 30
step_lengths = np.linspace(1, limit, 1000)

plt.figure(figsize=(LATEX_TEXTWIDTH, LATEX_TEXTWIDTH))

for mu, color in zip(mu_values, blues):
    pdf_power = step_lengths**(-(mu))
    plt.plot(step_lengths, pdf_power, color=color, label=f"Power Law ($\mu$={mu:.2f})")

for alpha, color in zip(alpha_values, reds):
    pdf_exp = alpha*np.exp(-alpha*step_lengths)
    plt.plot(step_lengths, pdf_exp, color=color, label=f"Exponential ($\\alpha$={alpha})")

pdf_gauss = norm.pdf(step_lengths, loc=1, scale=1)
plt.plot(step_lengths, pdf_gauss, color='green', label="Gaussian")

# plt.yscale('log')
plt.xlim(1, limit)
plt.ylim(0, 0.1)
plt.grid(ls="--", linewidth=0.4)
plt.xlabel('Step Length')
plt.ylabel('Probability Density')
legend = plt.legend(loc='upper right', fancybox=True, edgecolor='gray', borderpad=0.3, facecolor='white', frameon=True, framealpha=1)
legend.get_frame().set_linewidth(0.5)

plt.tight_layout()
plt.savefig(Path(__file__).parent.parent / 'resources/step_length_probability.pdf', format='pdf')