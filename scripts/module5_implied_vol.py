"""
Module 5 demo: implied volatility from REAL US option prices.
Run from the project folder:   python scripts/module5_implied_vol.py

Needs internet. Change TICKER to analyse a different US stock or ETF.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import call_price
from src.implied_vol import implied_vol, newton_implied_vol
from src import market_data as md

TICKER = "SPY"        # S&P 500 ETF: the most-traded options in the world
DIV_YIELD = 0.012     # SPY pays ~1.2% a year in dividends (use ~0.005 for AAPL, 0 for most growth stocks)
TARGET_DAYS = 30      # analyse the expiry closest to ~1 month out

# ---- 1. Warm-up: can we recover a volatility we already know? ----------
print("WARM-UP: price an option at vol=20%, then ask the solver to find the vol")
known_price = call_price(100, 100, 1.0, 0.05, 0.20)
sigma, guesses = newton_implied_vol(known_price, 100, 100, 1.0, 0.05, "call")
print(f"  Price = {known_price:.4f}.  Newton's guesses: "
      + " -> ".join(f"{g:.4f}" for g in guesses))
print(f"  Recovered vol = {sigma:.6f}  (Brent agrees: {implied_vol(known_price, 100, 100, 1.0, 0.05):.6f})\n")

# ---- 2. Download real market data --------------------------------------
print(f"Downloading {TICKER} data from Yahoo Finance...")
try:
    ticker = md.get_ticker(TICKER)
    S = md.get_spot(ticker)
    r = md.get_risk_free_rate()
    expiry = md.pick_expiry(ticker, TARGET_DAYS)
    raw = md.get_chain(ticker, expiry)
    rv_1y = md.realized_vol(ticker, "1y")
    rv_1m = md.realized_vol(ticker, "1mo")
except Exception as e:
    print(f"\nCouldn't download data ({type(e).__name__}: {e}).")
    print("Check your internet connection, or that `pip install yfinance` worked, and try again.")
    sys.exit(1)

T = raw["T"].iloc[0]
print(f"  Spot price      : ${S:.2f}")
print(f"  Risk-free rate  : {r:.2%}  (13-week T-bill)")
print(f"  Expiry analysed : {expiry}  ({T * 365:.0f} days away)")
print(f"  Options in chain: {len(raw)} -> ", end="")

chain = md.clean_chain(raw, S)
chain = md.add_implied_vols(chain, S, r, DIV_YIELD)
smile = md.out_of_the_money(chain, S)
print(f"{len(chain)} usable after cleaning")

os.makedirs("data", exist_ok=True)
csv_path = f"data/{TICKER}_{expiry}_chain.csv"
chain.to_csv(csv_path, index=False)
print(f"  Saved cleaned data to {csv_path}\n")

# ---- 3. What is the market saying? -------------------------------------
atm_row = smile.iloc[(smile["strike"] - S).abs().argmin()]
atm_iv = atm_row["iv"]
print("WHAT THE MARKET EXPECTS")
print(f"  At-the-money implied vol : {atm_iv:.1%}   (market's forecast for the next {T * 365:.0f} days)")
print(f"  Realized vol, last month : {rv_1m:.1%}")
print(f"  Realized vol, last year  : {rv_1y:.1%}")
daily_move = S * atm_iv / np.sqrt(252)
print(f"  -> The market is pricing a typical daily move of about +/-${daily_move:.2f} "
      f"({atm_iv / np.sqrt(252):.2%})")

print("\nSample of the smile (out-of-the-money options):")
step = max(len(smile) // 12, 1)
print(smile.iloc[::step][["type", "strike", "mid", "iv", "yahoo_iv"]]
      .to_string(index=False, formatters={"iv": "{:.1%}".format, "yahoo_iv": "{:.1%}".format,
                                         "mid": "{:.2f}".format}))

# ---- 4. Charts ---------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

puts = smile[smile["type"] == "put"]
calls = smile[smile["type"] == "call"]
ax1.plot(puts["strike"], puts["iv"] * 100, "o-", markersize=3, label="OTM puts")
ax1.plot(calls["strike"], calls["iv"] * 100, "o-", markersize=3, label="OTM calls")
ax1.axvline(S, color="grey", linestyle=":", label=f"Spot ${S:.0f}")
ax1.axhline(rv_1m * 100, color="black", linestyle="--", linewidth=1, label=f"Realized vol, 1 month ({rv_1m:.1%})")
ax1.set_xlabel("Strike")
ax1.set_ylabel("Implied volatility (%)")
ax1.set_title(f"{TICKER} volatility smile, expiry {expiry}")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.scatter(chain["yahoo_iv"] * 100, chain["iv"] * 100, s=10, alpha=0.6)
lim = [0, max(chain["iv"].max(), chain["yahoo_iv"].max()) * 100 * 1.05]
ax2.plot(lim, lim, "k--", linewidth=1, label="Perfect agreement")
ax2.set_xlim(lim)
ax2.set_ylim(lim)
ax2.set_xlabel("Yahoo's implied vol (%)")
ax2.set_ylabel("Our implied vol (%)")
ax2.set_title("Sanity check: our solver vs Yahoo's numbers")
ax2.legend()
ax2.grid(alpha=0.3)

fig.tight_layout()
os.makedirs("images", exist_ok=True)
filename = f"images/module5_smile_{TICKER}.png"
fig.savefig(filename, dpi=150, bbox_inches="tight")
print(f"\nChart saved to {filename}")
