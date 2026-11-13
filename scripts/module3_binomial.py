"""
Module 3 demo: binomial trees, convergence to Black-Scholes, and American options.
Run from the project folder:   python scripts/module3_binomial.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.binomial import binomial_price
from src.black_scholes import call_price, put_price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

# ---- 1. Does the tree agree with Black-Scholes? ------------------------
bs_call = call_price(S, K, T, r, sigma)
print(f"Black-Scholes call: {bs_call:.4f}")
for N in [1, 2, 5, 10, 50, 200, 1000]:
    tree = binomial_price(S, K, T, r, sigma, N, "call")
    print(f"  Tree with N={N:<5} steps: {tree:.4f}   (error {tree - bs_call:+.4f})")

# ---- 2. American vs European ------------------------------------------
print("\nAmerican vs European (N=500 steps):")
for opt in ["call", "put"]:
    euro = binomial_price(S, K, T, r, sigma, 500, opt, american=False)
    amer = binomial_price(S, K, T, r, sigma, 500, opt, american=True)
    print(f"  {opt:<4}  European={euro:.4f}  American={amer:.4f}  "
          f"early-exercise premium={amer - euro:.4f}")
print("  -> The American call is worth the SAME as the European one (no dividends),")
print("     but the American put is worth MORE: sometimes it pays to exercise early.")

# ---- 3. Charts ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: convergence
steps = np.arange(1, 151)
tree_prices = [binomial_price(S, K, T, r, sigma, n, "call") for n in steps]
ax1.plot(steps, tree_prices, label="Binomial tree")
ax1.axhline(bs_call, color="black", linestyle="--", label="Black-Scholes")
ax1.set_xlabel("Number of steps N")
ax1.set_ylabel("Call price")
ax1.set_title("Tree price converges to Black-Scholes as N grows")
ax1.legend()
ax1.grid(alpha=0.3)

# Right: early-exercise premium of a put across stock prices
stock_prices = np.linspace(60, 140, 81)
euro_put = [binomial_price(s, K, T, r, sigma, 300, "put") for s in stock_prices]
amer_put = [binomial_price(s, K, T, r, sigma, 300, "put", american=True) for s in stock_prices]
ax2.plot(stock_prices, amer_put, label="American put")
ax2.plot(stock_prices, euro_put, label="European put")
ax2.plot(stock_prices, np.maximum(K - stock_prices, 0), "k--", label="Exercise now")
ax2.set_xlabel("Stock price")
ax2.set_ylabel("Put value")
ax2.set_title("American put >= European put (deep in the money, exercise early)")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
os.makedirs("images", exist_ok=True)
filename = f"images/module3_binomial_K{K}.png"
fig.savefig(filename, dpi=150, bbox_inches="tight")
print(f"\nChart saved to {filename}")
