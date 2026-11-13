"""
Module 1 demo: price some options and see how price changes with the stock.
Run from the project folder:   python scripts/module1_demo.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Let this script find the 'src' folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import call_price, put_price, put_call_parity_gap

# ---- 1. Price one option ----------------------------------------------
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
print(f"Stock={S}, Strike={K}, T={T}yr, r={r:.0%}, vol={sigma:.0%}")
print(f"  Call price: {call_price(S, K, T, r, sigma):.4f}")
print(f"  Put price : {put_price(S, K, T, r, sigma):.4f}")
print(f"  Put-call parity gap (should be ~0): {put_call_parity_gap(S, K, T, r, sigma):.2e}")

# ---- 2. What happens when volatility goes up? -------------------------
print("\nVolatility vs call price (more uncertainty = more valuable option):")
for vol in [0.10, 0.20, 0.40, 0.80]:
    print(f"  vol={vol:.0%}  ->  call={call_price(S, K, T, r, vol):.2f}")

# ---- 3. Chart: option price vs stock price ----------------------------
stock_prices = np.linspace(50, 150, 200)
fig, ax = plt.subplots(figsize=(8, 5))
for t in [1.0, 0.25, 0.01]:
    ax.plot(stock_prices, call_price(stock_prices, K, t, r, sigma), label=f"Call, T={t}yr")
ax.plot(stock_prices, np.maximum(stock_prices - K, 0), "k--", label="Payoff at expiry")
ax.set_xlabel("Stock price")
ax.set_ylabel("Option value")
ax.set_title("Black-Scholes call value: time value melts into the payoff as expiry nears")
ax.legend()
ax.grid(alpha=0.3)

os.makedirs("images", exist_ok=True)
fig.savefig(f"images/module1_price_curves_K{K}.png", dpi=150, bbox_inches="tight")
print(f"\nChart saved to images/module1_price_curves_K{K}.png")
