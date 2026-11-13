"""Tests for Monte Carlo pricing. Run with:  python -m pytest"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import price
from src.monte_carlo import mc_asian, mc_european

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20


@pytest.mark.parametrize("opt", ["call", "put"])
def test_mc_within_confidence_interval_of_black_scholes(opt):
    mc, se = mc_european(S, K, T, r, sigma, 400_000, opt, seed=123)
    assert abs(mc - price(S, K, T, r, sigma, opt)) < 4 * se


def test_antithetic_reduces_error():
    _, plain_se = mc_european(S, K, T, r, sigma, 100_000, seed=5)
    _, anti_se = mc_european(S, K, T, r, sigma, 100_000, antithetic=True, seed=5)
    assert anti_se < plain_se


def test_same_seed_gives_same_answer():
    assert mc_european(S, K, T, r, sigma, 10_000, seed=9) == mc_european(S, K, T, r, sigma, 10_000, seed=9)


def test_asian_call_cheaper_than_european_call():
    asian, _ = mc_asian(S, K, T, r, sigma, 20_000, 100, seed=11)
    assert asian < price(S, K, T, r, sigma, "call")
