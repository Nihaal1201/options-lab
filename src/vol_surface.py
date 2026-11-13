"""
The implied volatility SURFACE.

Intuition
---------
Module 5 gave you one smile: implied vol across strikes, for ONE expiry.
Repeat that for many expiries and stack the smiles one behind another, and you
get a landscape, the volatility surface:

    x-axis : moneyness  (strike / stock price; 1.0 = at the money)
    y-axis : days to expiry
    height : implied volatility

Traders read two things off it:
- SKEW (across strikes): how much more the market charges for crash
  protection (low strikes) than for upside bets.
- TERM STRUCTURE (across time): is short-term fear higher or lower than
  long-term fear? Normally longer expiries carry slightly higher vol
  ('upward sloping'). When short-dated vol spikes above long-dated vol
  ('inverted'), the market is scared about something happening NOW.
"""

import time
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from src import market_data as md


def days_until(expiry, today=None):
    today = today or date.today()
    return (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days


def choose_expiries(expiries, min_days=7, max_days=365, n=10, today=None):
    """
    Pick about n expiries spread evenly between min_days and max_days.
    (SPY lists an expiry almost every day; we don't need all of them.)
    """
    eligible = [e for e in expiries if min_days <= days_until(e, today) <= max_days]
    if len(eligible) <= n:
        return eligible
    idx = np.unique(np.linspace(0, len(eligible) - 1, n).round().astype(int))
    return [eligible[i] for i in idx]


def download_surface_data(ticker, S, r, q, expiries, moneyness_range=(0.7, 1.3), pause=0.3):
    """Download, clean and compute implied vols for every chosen expiry."""
    frames = []
    for expiry in expiries:
        try:
            raw = md.get_chain(ticker, expiry)
        except Exception as e:
            print(f"  skipped {expiry} ({type(e).__name__})")
            continue
        chain = md.clean_chain(raw, S, moneyness_range=moneyness_range)
        chain = md.add_implied_vols(chain, S, r, q)
        chain = md.out_of_the_money(chain, S)
        chain["days"] = days_until(expiry)
        print(f"  {expiry}  ({chain['days'].iloc[0] if len(chain) else '?':>3} days): {len(chain)} options")
        frames.append(chain)
        time.sleep(pause)   # be polite to Yahoo's servers
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def term_structure(surface_df):
    """
    For each expiry: at-the-money vol, plus a simple skew measure
    (vol at 90% moneyness minus vol at 100%: how much extra the market
    charges for protection against a 10% drop).
    """
    rows = []
    for (expiry, days), grp in surface_df.groupby(["expiry", "days"]):
        grp = grp.sort_values("moneyness")
        m, iv = grp["moneyness"].to_numpy(), grp["iv"].to_numpy()
        if len(grp) < 3 or m.min() > 0.9 or m.max() < 1.0:
            continue
        atm = np.interp(1.0, m, iv)
        rows.append({"expiry": expiry, "days": days, "atm_iv": atm,
                     "skew_90_100": np.interp(0.9, m, iv) - atm})
    return pd.DataFrame(rows).sort_values("days").reset_index(drop=True)


def interpolate_grid(surface_df, n_moneyness=50, n_days=40):
    """
    Turn scattered (moneyness, days, iv) points into a smooth regular grid
    so it can be drawn as a surface. Points outside the data stay NaN
    (we never invent data where the market gives us none).
    """
    m_grid = np.linspace(surface_df["moneyness"].min(), surface_df["moneyness"].max(), n_moneyness)
    d_grid = np.linspace(surface_df["days"].min(), surface_df["days"].max(), n_days)
    M, D = np.meshgrid(m_grid, d_grid)
    # Scale days to a similar range as moneyness so 'nearest' is meaningful
    scale = surface_df["days"].max()
    points = np.column_stack([surface_df["moneyness"], surface_df["days"] / scale])
    Z = griddata(points, surface_df["iv"], (M, D / scale), method="linear")
    return M, D, Z
