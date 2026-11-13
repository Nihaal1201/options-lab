"""
Black-Scholes pricing for European options.

Intuition
---------
A call option is a bet that the stock ends ABOVE the strike K at expiry.
Its fair value today = (probability-weighted payoff) discounted back to today.

    Call = S * N(d1)  -  K * e^(-rT) * N(d2)
           ^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^
           what you get   what you pay (the strike, discounted to today),
           (the stock)    weighted by the risk-neutral chance you exercise

- N(d2) = risk-neutral probability the option finishes in the money.
- N(d1) = the option's delta: how many shares behave like the option.

Inputs (all functions use the same names):
    S     : current stock price
    K     : strike price
    T     : time to expiry in YEARS (e.g. 30 days -> 30/365)
    r     : risk-free rate, annual, continuous (e.g. 0.05 for 5%)
    sigma : volatility, annual (e.g. 0.20 for 20%)
"""

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    """The two 'standardised distances' between the stock and the strike."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma):
    """Price of a European call option."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    """Price of a European put option (a bet the stock ends BELOW K)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def price(S, K, T, r, sigma, option_type="call"):
    """Convenience wrapper: option_type is 'call' or 'put'."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma)
    if option_type == "put":
        return put_price(S, K, T, r, sigma)
    raise ValueError("option_type must be 'call' or 'put'")


def put_call_parity_gap(S, K, T, r, sigma):
    """
    Sanity check. For European options:  C - P = S - K * e^(-rT)
    (A call minus a put with the same strike behaves exactly like owning
    the stock and owing K at expiry.) This should return ~0.
    """
    lhs = call_price(S, K, T, r, sigma) - put_price(S, K, T, r, sigma)
    rhs = S - K * np.exp(-r * T)
    return lhs - rhs
