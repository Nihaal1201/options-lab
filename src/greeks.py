"""
The Greeks: how sensitive an option's price is to each input.

Intuition (think of driving a car)
---------------------------------
- Delta : speed.  If the stock moves +$1, how much does the option move?
- Gamma : acceleration.  How fast does delta itself change as the stock moves?
- Vega  : sensitivity to volatility.  If vol rises 1 point (20% -> 21%), price change?
- Theta : time decay.  How much value does the option lose per day, all else equal?
- Rho   : sensitivity to interest rates.  If rates rise 1 point, price change?

Units used here (the way traders quote them):
- delta, gamma : per $1 move in the stock
- vega         : per 1 volatility point (0.01)
- theta        : per calendar day
- rho          : per 1 interest-rate point (0.01)
"""

import numpy as np
from scipy.stats import norm

from src.black_scholes import _d1_d2


def delta(S, K, T, r, sigma, option_type="call"):
    """Call delta is between 0 and 1; put delta is between -1 and 0."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """Same for calls and puts. Largest when the stock is near the strike."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    """Same for calls and puts. Price change for a 1-point rise in volatility."""
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def theta(S, K, T, r, sigma, option_type="call"):
    """Price change per day that passes (usually negative: options 'melt')."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    decay = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        carry = -r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        carry = r * K * np.exp(-r * T) * norm.cdf(-d2)
    return (decay + carry) / 365


def rho(S, K, T, r, sigma, option_type="call"):
    """Price change for a 1-point rise in interest rates."""
    _, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def all_greeks(S, K, T, r, sigma, option_type="call"):
    """Return every Greek in one dictionary."""
    return {
        "delta": delta(S, K, T, r, sigma, option_type),
        "gamma": gamma(S, K, T, r, sigma),
        "vega": vega(S, K, T, r, sigma),
        "theta": theta(S, K, T, r, sigma, option_type),
        "rho": rho(S, K, T, r, sigma, option_type),
    }
