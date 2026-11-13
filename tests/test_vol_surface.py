"""Tests for the volatility surface (no internet needed). Run with:  python -m pytest"""

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import vol_surface as vs

TODAY = date(2026, 1, 1)


def expiry(days):
    return (TODAY + timedelta(days=days)).isoformat()


def fake_surface():
    """A surface with a known shape: skewed smile + rising term structure."""
    rows = []
    for days in [10, 30, 90, 180]:
        for m in np.linspace(0.8, 1.2, 17):
            iv = 0.15 + 0.02 * days / 180 - 0.3 * (m - 1)
            rows.append({"expiry": expiry(days), "days": days, "moneyness": m, "iv": iv})
    return pd.DataFrame(rows)


def test_choose_expiries_respects_window_and_count():
    listed = [expiry(d) for d in range(1, 800)]
    chosen = vs.choose_expiries(listed, min_days=7, max_days=365, n=10, today=TODAY)
    days = [vs.days_until(e, TODAY) for e in chosen]
    assert len(chosen) == 10
    assert min(days) == 7 and max(days) == 365
    assert days == sorted(days)


def test_choose_expiries_keeps_all_when_few():
    listed = [expiry(d) for d in (3, 20, 60, 400)]
    assert vs.choose_expiries(listed, 7, 365, 10, today=TODAY) == [expiry(20), expiry(60)]


def test_term_structure_reads_known_shape():
    ts = vs.term_structure(fake_surface())
    assert list(ts["days"]) == [10, 30, 90, 180]
    assert ts["atm_iv"].is_monotonic_increasing                         # upward sloping
    assert np.allclose(ts["skew_90_100"], 0.03)                         # 0.3 * 10% drop


def test_interpolated_grid_matches_data():
    df = fake_surface()
    M, D, Z = vs.interpolate_grid(df)
    expected = 0.15 + 0.02 * D / 180 - 0.3 * (M - 1)
    inside = ~np.isnan(Z)
    assert inside.mean() > 0.9
    assert np.allclose(Z[inside], expected[inside], atol=1e-6)          # a plane is interpolated exactly
