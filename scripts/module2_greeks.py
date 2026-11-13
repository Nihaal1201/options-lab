"""
Module 2 demo: compute the Greeks and see how they change with the stock price.
Run from the project folder:   python scripts/module2_greeks.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import call_price
from src.greeks import all_greeks, delta, gamma, theta, vega

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

# ---- 1. The Greeks for one option -------------------------------------
print(f"Stock={S}, Strike={K}, T={T}yr, r={r:.0%}, vol={sigma:.0%}\n")
for option_type in ["call", "put"]:
    print(f"{option_type.upper()} Greeks:")
    for name, value in all_greeks(S, K, T, r, sigma, option_type).items():
        print(f"  {name:<6} {value:+.4f}")
    print()

# ---- 2. Check that delta really predicts the price move ---------------
d = delta(S, K, T, r, sigma)
before = call_price(S, K, T, r, sigma)
after = call_price(S + 1, K, T, r, sigma)
print("Does delta predict a $1 stock move?")
print(f"  Delta says the call moves by: {d:.4f}")
print(f"  Actual move (repriced)      : {after - before:.4f}")
print("  (The small gap is gamma: delta itself changes as the stock moves.)\n")

# ---- 3. Chart: each Greek vs stock price, for different expiries ------
stock_prices = np.linspace(60, 140, 300)
panels = [
    ("Delta (call)", lambda s, t: delta(s, K, t, r, sigma)),
    ("Gamma", lambda s, t: gamma(s, K, t, r, sigma)),
    ("Vega (per 1 vol pt)", lambda s, t: vega(s, K, t, r, sigma)),
    ("Theta (call, per day)", lambda s, t: theta(s, K, t, r, sigma)),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, (title, func) in zip(axes.flat, panels):
    for t in [1.0, 0.25, 0.05]:
        ax.plot(stock_prices, func(stock_prices, t), label=f"T={t}yr")
    ax.axvline(K, color="grey", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Stock price")
    ax.grid(alpha=0.3)
axes[0, 0].legend()
fig.suptitle(f"Black-Scholes Greeks (K={K}, vol={sigma:.0%}): action concentrates at the strike near expiry")
fig.tight_layout()

os.makedirs("images", exist_ok=True)
filename = f"images/module2_greeks_K{K}.png"
fig.savefig(filename, dpi=150, bbox_inches="tight")
print(f"Chart saved to {filename}")
