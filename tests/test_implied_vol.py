"""Tests for implied volatility and data cleaning (no internet needed). Run with:  python -m pytest"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import price
from src.implied_vol import implied_vol, newton_implied_vol
from src.market_data import add_implied_vols, clean_chain, out_of_the_money


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("K", [80, 100, 120])
@pytest.mark.parametrize("vol", [0.1, 0.25, 0.6])
def test_round_trip(opt, K, vol):
    p = price(100, K, 0.5, 0.04, vol, opt)
    assert implied_vol(p, 100, K, 0.5, 0.04, opt) == pytest.approx(vol, abs=1e-6)


def test_newton_matches_brent():
    p = price(100, 105, 0.25, 0.03, 0.33, "call")
    newton, _ = newton_implied_vol(p, 100, 105, 0.25, 0.03, "call")
    assert newton == pytest.approx(implied_vol(p, 100, 105, 0.25, 0.03), abs=1e-6)


def test_dividend_round_trip():
    S, q, T = 100, 0.02, 1.0
    p = price(S * np.exp(-q * T), 100, T, 0.05, 0.3, "put")
    assert implied_vol(p, S, 100, T, 0.05, "put", q=q) == pytest.approx(0.3, abs=1e-6)


def test_price_below_intrinsic_gives_nan():
    # A call with strike 80 on a $100 stock is worth at least ~$20; $5 is impossible
    assert np.isnan(implied_vol(5.0, 100, 80, 0.5, 0.04, "call"))


def test_cleaning_and_otm_selection():
    S = 100
    rows = []
    for K in [70, 90, 100, 110, 130]:
        for opt in ["call", "put"]:
            p = price(S, K, 0.1, 0.04, 0.2, opt)
            rows.append({"type": opt, "expiry": "x", "T": 0.1, "strike": K, "bid": p * 0.98,
                         "ask": p * 1.02, "last": p, "volume": 5, "openInterest": 100, "yahoo_iv": 0.2})
    df = pd.DataFrame(rows)
    cleaned = add_implied_vols(clean_chain(df, S), S, 0.04)
    assert cleaned["strike"].between(75, 125).all()      # far strikes removed
    assert np.allclose(cleaned["iv"], 0.2, atol=1e-3)    # mid prices recover the vol
    otm = out_of_the_money(cleaned, S)
    assert (otm[otm["type"] == "put"]["strike"] < S).all()
    assert (otm[otm["type"] == "call"]["strike"] >= S).all()
