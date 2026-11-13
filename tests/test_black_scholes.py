"""Tests: prove the pricer matches textbook values. Run with:  python -m pytest"""

import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.black_scholes import call_price, put_price, put_call_parity_gap


def test_textbook_call():
    # Standard example: S=100, K=100, T=1, r=5%, vol=20% -> call ~ 10.4506
    assert call_price(100, 100, 1, 0.05, 0.20) == pytest.approx(10.4506, abs=1e-4)


def test_textbook_put():
    assert put_price(100, 100, 1, 0.05, 0.20) == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity():
    for S in [80, 100, 120]:
        assert abs(put_call_parity_gap(S, 100, 0.5, 0.03, 0.3)) < 1e-10


def test_higher_vol_means_higher_price():
    assert call_price(100, 100, 1, 0.05, 0.40) > call_price(100, 100, 1, 0.05, 0.20)
