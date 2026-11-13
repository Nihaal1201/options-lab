"""
Tests for the Greeks. The idea: a Greek is just a slope, so we can check each
formula by nudging one input a tiny bit and repricing (a 'finite difference').
Run with:  python -m pytest
"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import price
from src.greeks import delta, gamma, rho, theta, vega

S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
h = 1e-4  # a tiny nudge


@pytest.mark.parametrize("opt", ["call", "put"])
def test_delta_matches_slope(opt):
    slope = (price(S + h, K, T, r, sigma, opt) - price(S - h, K, T, r, sigma, opt)) / (2 * h)
    assert delta(S, K, T, r, sigma, opt) == pytest.approx(slope, abs=1e-6)


def test_gamma_matches_curvature():
    up, mid, down = (price(s, K, T, r, sigma) for s in (S + 0.01, S, S - 0.01))
    curvature = (up - 2 * mid + down) / 0.01**2
    assert gamma(S, K, T, r, sigma) == pytest.approx(curvature, abs=1e-5)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_vega_matches_slope(opt):
    slope = (price(S, K, T, r, sigma + h, opt) - price(S, K, T, r, sigma - h, opt)) / (2 * h)
    assert vega(S, K, T, r, sigma) == pytest.approx(slope / 100, abs=1e-6)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_theta_matches_slope(opt):
    # Theta is the value lost as time-to-expiry SHRINKS, hence the minus sign
    slope = -(price(S, K, T + h, r, sigma, opt) - price(S, K, T - h, r, sigma, opt)) / (2 * h)
    assert theta(S, K, T, r, sigma, opt) == pytest.approx(slope / 365, abs=1e-6)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_rho_matches_slope(opt):
    slope = (price(S, K, T, r + h, sigma, opt) - price(S, K, T, r - h, sigma, opt)) / (2 * h)
    assert rho(S, K, T, r, sigma, opt) == pytest.approx(slope / 100, abs=1e-6)


def test_textbook_call_delta():
    assert delta(S, K, T, r, sigma, "call") == pytest.approx(0.6368, abs=1e-4)
