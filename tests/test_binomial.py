"""Tests for the binomial tree. Run with:  python -m pytest"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.binomial import binomial_price
from src.black_scholes import price

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20


@pytest.mark.parametrize("opt", ["call", "put"])
def test_converges_to_black_scholes(opt):
    tree = binomial_price(S, K, T, r, sigma, 2000, opt)
    assert tree == pytest.approx(price(S, K, T, r, sigma, opt), abs=0.01)


def test_american_call_equals_european_without_dividends():
    euro = binomial_price(S, K, T, r, sigma, 500, "call")
    amer = binomial_price(S, K, T, r, sigma, 500, "call", american=True)
    assert amer == pytest.approx(euro, abs=1e-10)


def test_american_put_worth_more():
    euro = binomial_price(S, K, T, r, sigma, 500, "put")
    amer = binomial_price(S, K, T, r, sigma, 500, "put", american=True)
    assert amer > euro


def test_american_put_never_below_exercise_value():
    # Deep in the money: an American put is worth at least K - S
    assert binomial_price(60, K, T, r, sigma, 500, "put", american=True) >= K - 60
