"""
Module 4 demo: Monte Carlo pricing, confidence intervals, variance reduction,
and an exotic (Asian) option that has no formula.
Run from the project folder:   python scripts/module4_monte_carlo.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import call_price
from src.monte_carlo import mc_asian, mc_european, simulate_paths

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
bs = call_price(S, K, T, r, sigma)

# ---- 1. More simulations = more accurate --------------------------------
print(f"Black-Scholes call: {bs:.4f}\n")
print("Monte Carlo with more and more simulated universes:")
for n in [100, 1_000, 10_000, 100_000, 1_000_000]:
    price, se = mc_european(S, K, T, r, sigma, n, seed=42)
    print(f"  {n:>9,} paths: {price:.4f}  95% CI [{price - 1.96*se:.4f}, {price + 1.96*se:.4f}]")
print("  -> 100x more paths makes the error ~10x smaller (error shrinks like 1/sqrt(N)).")

# ---- 2. Variance reduction: antithetic variates -----------------------
plain_p, plain_se = mc_european(S, K, T, r, sigma, 100_000, seed=1)
anti_p, anti_se = mc_european(S, K, T, r, sigma, 100_000, antithetic=True, seed=1)
print("\nSame 100,000 paths, smarter sampling:")
print(f"  Plain      : {plain_p:.4f}  (std error {plain_se:.4f})")
print(f"  Antithetic : {anti_p:.4f}  (std error {anti_se:.4f})")
print(f"  -> about {(plain_se / anti_se)**2:.1f}x fewer paths needed for the same accuracy")

# ---- 3. An option with NO formula: the Asian call ---------------------
asian_p, asian_se = mc_asian(S, K, T, r, sigma, 50_000, 252, seed=7)
print("\nAsian call (payoff uses the AVERAGE price over the year):")
print(f"  Price: {asian_p:.4f} +/- {1.96*asian_se:.4f}   vs regular call {bs:.4f}")
print("  -> Cheaper, because an average swings less than a single final price.")

# ---- 4. Charts --------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: 50 simulated stock paths
rng = np.random.default_rng(0)
paths = simulate_paths(S, T, r, sigma, 50, 252, rng)
days = np.linspace(0, T, 253)
ax1.plot(days, paths.T, linewidth=0.8, alpha=0.7)
ax1.axhline(K, color="black", linestyle="--", label=f"Strike K={K}")
ax1.set_xlabel("Time (years)")
ax1.set_ylabel("Stock price")
ax1.set_title(f"50 simulated futures for the stock (vol={sigma:.0%})")
ax1.legend()
ax1.grid(alpha=0.3)

# Right: running estimate converging to Black-Scholes
rng = np.random.default_rng(3)
Z = rng.standard_normal(200_000)
ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
disc_payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
n = np.arange(1, len(Z) + 1)
running_mean = np.cumsum(disc_payoffs) / n
running_mean_sq = np.cumsum(disc_payoffs**2) / n
running_se = np.sqrt(np.maximum(running_mean_sq - running_mean**2, 0) / n)
start = 30  # the band is meaningless with only a handful of paths
ax2.plot(n, running_mean, label="Monte Carlo estimate")
ax2.fill_between(n[start:], running_mean[start:] - 1.96 * running_se[start:],
                 running_mean[start:] + 1.96 * running_se[start:], alpha=0.25, label="95% confidence band")
ax2.axhline(bs, color="black", linestyle="--", label="Black-Scholes")
ax2.set_xscale("log")
ax2.set_ylim(bs - 2, bs + 2)
ax2.set_xlabel("Number of simulated paths (log scale)")
ax2.set_ylabel("Call price")
ax2.set_title("Estimate tightens around the true price as paths increase")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
os.makedirs("images", exist_ok=True)
filename = f"images/module4_monte_carlo_K{K}.png"
fig.savefig(filename, dpi=150, bbox_inches="tight")
print(f"\nChart saved to {filename}")
