"""Tests for market-data helpers using fake Yahoo responses (no internet needed)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import market_data as md


class FakeTicker:
    def __init__(self, last_price, closes):
        self._last, self._closes = last_price, closes

    @property
    def fast_info(self):
        if self._last is None:
            raise KeyError("last_price")
        return {"last_price": self._last}

    def history(self, period="5d"):
        return pd.DataFrame({"Close": self._closes})


def test_spot_prefers_live_last_price():
    assert md.get_spot(FakeTicker(757.0, [750.0, 755.0])) == 757.0


def test_spot_skips_blank_row_after_hours():
    # The bug we hit: a blank (nan) row for today used to make spot = nan
    assert md.get_spot(FakeTicker(None, [750.0, 755.0, np.nan])) == 755.0
    assert md.get_spot(FakeTicker(np.nan, [750.0, 755.0, np.nan])) == 755.0


def test_spot_raises_when_no_price_at_all():
    with pytest.raises(ValueError):
        md.get_spot(FakeTicker(None, [np.nan, np.nan]))


def test_rate_skips_blank_row(monkeypatch):
    monkeypatch.setattr(md, "get_ticker", lambda s: FakeTicker(None, [4.1, 4.0, np.nan]))
    assert md.get_risk_free_rate() == pytest.approx(0.04)


def test_rate_falls_back_to_default(monkeypatch):
    monkeypatch.setattr(md, "get_ticker", lambda s: FakeTicker(None, [np.nan]))
    assert md.get_risk_free_rate(default=0.035) == 0.035
