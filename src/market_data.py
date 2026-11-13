"""
Download real US option data from Yahoo Finance (via the free yfinance package)
and clean it up for analysis.

Real market data is messy, so this file does the boring-but-crucial work:
- picks a sensible price for each option (mid of bid/ask, or last trade)
- throws away illiquid or broken quotes
- converts expiry dates into T (years)
"""

from datetime import date, datetime

import numpy as np
import pandas as pd

from src.implied_vol import implied_vol


def get_ticker(symbol):
    import yfinance as yf   # imported here so the rest of the project works without it
    return yf.Ticker(symbol)


def get_spot(ticker):
    """
    Latest stock price.
    Tries Yahoo's live 'last price' first. If that's missing, uses the most
    recent daily close, skipping empty rows (outside market hours Yahoo often
    adds a blank row for today, which would otherwise give 'nan').
    """
    try:
        price = float(ticker.fast_info["last_price"])
        if np.isfinite(price) and price > 0:
            return price
    except Exception:
        pass
    closes = ticker.history(period="5d")["Close"].dropna()
    if closes.empty:
        raise ValueError("no recent price found for this ticker")
    return float(closes.iloc[-1])


def get_risk_free_rate(default=0.04):
    """13-week US Treasury bill yield (Yahoo symbol ^IRX), as a decimal."""
    try:
        irx = get_ticker("^IRX").history(period="5d")["Close"].dropna().iloc[-1]
        rate = float(irx) / 100
        return rate if np.isfinite(rate) else default
    except Exception:
        return default


def realized_vol(ticker, period="1y"):
    """
    Historical (realized) volatility: how much the stock ACTUALLY moved.
    Annualised standard deviation of daily log returns.
    """
    closes = ticker.history(period=period)["Close"]
    log_returns = np.log(closes / closes.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


def years_to_expiry(expiry, today=None):
    today = today or date.today()
    days = (datetime.strptime(expiry, "%Y-%m-%d").date() - today).days
    return max(days, 1) / 365


def pick_expiry(ticker, target_days=30):
    """The listed expiry closest to target_days from today."""
    today = date.today()
    expiries = ticker.options
    return min(expiries, key=lambda e: abs(
        (datetime.strptime(e, "%Y-%m-%d").date() - today).days - target_days))


def get_chain(ticker, expiry):
    """Calls and puts for one expiry, stacked into one tidy table."""
    chain = ticker.option_chain(expiry)
    frames = []
    for option_type, df in [("call", chain.calls), ("put", chain.puts)]:
        df = df.copy()
        df["type"] = option_type
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"lastPrice": "last", "impliedVolatility": "yahoo_iv"})
    out["expiry"] = expiry
    out["T"] = years_to_expiry(expiry)
    keep = ["type", "expiry", "T", "strike", "bid", "ask", "last",
            "volume", "openInterest", "yahoo_iv"]
    return out[keep]


def clean_chain(df, S, moneyness_range=(0.75, 1.25), max_rel_spread=0.5):
    """
    Keep only quotes we can trust.
    - price = midpoint of bid and ask when both exist (markets open),
      otherwise the last traded price (e.g. at weekends)
    - drop strikes far from the stock price
    - drop options nobody holds, and quotes with huge bid-ask spreads
    """
    df = df.copy()
    has_quotes = (df["bid"] > 0) & (df["ask"] > 0)
    df["mid"] = np.where(has_quotes, (df["bid"] + df["ask"]) / 2, df["last"])
    df["rel_spread"] = np.where(has_quotes, (df["ask"] - df["bid"]) / df["mid"], 0.0)

    lo, hi = moneyness_range
    mask = (
        (df["strike"] >= lo * S) & (df["strike"] <= hi * S)
        & (df["mid"] > 0.05)
        & (df["openInterest"].fillna(0) >= 10)
        & (df["rel_spread"] <= max_rel_spread)
    )
    return df[mask].reset_index(drop=True)


def add_implied_vols(df, S, r, q=0.0):
    """Compute our own implied vol for every row."""
    df = df.copy()
    df["iv"] = [implied_vol(row.mid, S, row.strike, row.T, r, row.type, q)
                for row in df.itertuples()]
    df["moneyness"] = df["strike"] / S
    return df.dropna(subset=["iv"]).reset_index(drop=True)


def out_of_the_money(df, S):
    """
    Standard practice: use puts below the stock price and calls above it.
    Out-of-the-money options are the most traded and their prices are almost
    pure 'volatility value', so their implied vols are the most reliable.
    """
    otm = ((df["type"] == "put") & (df["strike"] < S)) | ((df["type"] == "call") & (df["strike"] >= S))
    return df[otm].sort_values("strike").reset_index(drop=True)
