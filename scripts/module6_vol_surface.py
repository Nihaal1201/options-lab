"""
Module 6 demo: the 3D implied volatility surface from REAL US option prices.
Run from the project folder:   python scripts/module6_vol_surface.py

Needs internet (unless LOAD_CSV points at a file you saved earlier).
Produces:
  images/module6_surface_<TICKER>.png   static picture for your README
  images/module6_surface_<TICKER>.html  interactive 3D chart: open it in a browser and rotate it
"""

import os
import sys
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import market_data as md
from src import vol_surface as vs

TICKER = "SPY"
DIV_YIELD = 0.012       # SPY ~1.2% (AAPL ~0.005, most growth stocks 0)
N_EXPIRIES = 10         # how many expiries to download (more = slower)
MAX_DAYS = 365          # furthest expiry to include
LOAD_CSV = None         # e.g. "data/SPY_surface_2026-11-12.csv" to re-plot saved data offline

# ---- 1. Get the data ----------------------------------------------------
if LOAD_CSV:
    surface = pd.read_csv(LOAD_CSV)
    S = surface.attrs.get("spot", float(surface["spot"].iloc[0]))
    print(f"Loaded {len(surface)} options from {LOAD_CSV}")
else:
    print(f"Downloading {TICKER} option chains from Yahoo Finance...")
    try:
        ticker = md.get_ticker(TICKER)
        S = md.get_spot(ticker)
        r = md.get_risk_free_rate()
        expiries = vs.choose_expiries(ticker.options, min_days=7, max_days=MAX_DAYS, n=N_EXPIRIES)
        print(f"  Spot ${S:.2f}, rate {r:.2%}, {len(expiries)} expiries chosen")
        surface = vs.download_surface_data(ticker, S, r, DIV_YIELD, expiries)
    except Exception as e:
        print(f"\nCouldn't download data ({type(e).__name__}: {e}).")
        print("Check your internet connection and try again.")
        sys.exit(1)
    if surface.empty:
        print("No usable options came back. Try again during US market hours.")
        sys.exit(1)
    surface["spot"] = S
    os.makedirs("data", exist_ok=True)
    csv_path = f"data/{TICKER}_surface_{date.today()}.csv"
    surface.to_csv(csv_path, index=False)
    print(f"  Saved {len(surface)} options to {csv_path}")

# ---- 2. Read the surface: term structure and skew -----------------------
ts = vs.term_structure(surface)
print("\nTERM STRUCTURE (one row per expiry)")
print(ts.to_string(index=False, formatters={
    "atm_iv": "{:.1%}".format, "skew_90_100": "{:+.1%}".format}))
if len(ts) >= 2:
    short, long = ts["atm_iv"].iloc[0], ts["atm_iv"].iloc[-1]
    shape = "UPWARD sloping (calm: more uncertainty further out)" if long > short \
        else "INVERTED (stressed: the market fears the near term most)"
    print(f"\n  Shortest {short:.1%} vs longest {long:.1%} -> term structure is {shape}")
    print(f"  Crash premium: a 10%-down put costs {ts['skew_90_100'].mean():.1%} more vol "
          f"than an at-the-money option, on average")

# ---- 3. Static chart for the README ------------------------------------
M, D, Z = vs.interpolate_grid(surface)

fig = plt.figure(figsize=(15, 6.5))
ax3d = fig.add_subplot(1, 2, 1, projection="3d")
ax3d.plot_surface(M, D, Z * 100, cmap="viridis", edgecolor="none", alpha=0.9)
ax3d.scatter(surface["moneyness"], surface["days"], surface["iv"] * 100, s=3, c="black", alpha=0.4)
ax3d.set_xlabel("Moneyness (strike / spot)")
ax3d.set_ylabel("Days to expiry")
ax3d.set_zlabel("Implied vol (%)")
ax3d.set_title(f"{TICKER} implied volatility surface ({date.today()}, spot ${S:.0f})")
ax3d.view_init(elev=22, azim=-55)

ax_smile = fig.add_subplot(2, 2, 2)
colors = plt.cm.viridis(np.linspace(0, 0.9, surface["days"].nunique()))
for color, (days, grp) in zip(colors, surface.groupby("days")):
    grp = grp.sort_values("moneyness")
    ax_smile.plot(grp["moneyness"], grp["iv"] * 100, color=color, linewidth=1.2, label=f"{days}d")
ax_smile.axvline(1.0, color="grey", linestyle=":")
ax_smile.set_xlabel("Moneyness")
ax_smile.set_ylabel("IV (%)")
ax_smile.set_title("Smiles flatten as expiry gets further away")
ax_smile.legend(fontsize=7, ncol=2)
ax_smile.grid(alpha=0.3)

ax_ts = fig.add_subplot(2, 2, 4)
ax_ts.plot(ts["days"], ts["atm_iv"] * 100, "o-", label="At-the-money IV")
ax_ts.plot(ts["days"], (ts["atm_iv"] + ts["skew_90_100"]) * 100, "s--", label="IV at 90% strike")
ax_ts.set_xlabel("Days to expiry")
ax_ts.set_ylabel("IV (%)")
ax_ts.set_title("Term structure: fear across time horizons")
ax_ts.legend()
ax_ts.grid(alpha=0.3)

fig.tight_layout()
os.makedirs("images", exist_ok=True)
png_path = f"images/module6_surface_{TICKER}.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"\nChart saved to {png_path}")

# ---- 4. Interactive 3D chart (rotate it with your mouse) ---------------
try:
    import plotly.graph_objects as go
    fig3d = go.Figure([
        go.Surface(x=M, y=D, z=Z * 100, colorscale="Viridis", opacity=0.9,
                   colorbar=dict(title="IV %"), name="Surface"),
        go.Scatter3d(x=surface["moneyness"], y=surface["days"], z=surface["iv"] * 100,
                     mode="markers", marker=dict(size=2, color="black"), name="Market quotes"),
    ])
    fig3d.update_layout(
        title=f"{TICKER} implied volatility surface ({date.today()}, spot ${S:.2f})",
        scene=dict(xaxis_title="Moneyness (K/S)", yaxis_title="Days to expiry", zaxis_title="Implied vol (%)"),
        height=750)
    html_path = f"images/module6_surface_{TICKER}.html"
    fig3d.write_html(html_path, include_plotlyjs="cdn")
    print(f"Interactive chart saved to {html_path}  (open it with:  open {html_path})")
except ImportError:
    print("Install plotly for the interactive chart:  pip install plotly")
