"""
Implied volatility: running Black-Scholes BACKWARDS.

Intuition
---------
So far: you choose sigma -> Black-Scholes gives you a price.
In the real market it's the other way round: the PRICE is visible on the
screen, and sigma is the unknown. Implied volatility is the sigma that makes
Black-Scholes reproduce the market price. It's the market's own forecast of
how wild the stock will be.

There's no formula to solve for sigma directly, so we SEARCH for it, like
guessing a number in "higher / lower":
- Newton's method: guess sigma, see how far the model price is from the market
  price, and use vega (the slope) to jump to a better guess. Very fast.
- Brent's method: trap the answer between a low and a high sigma and keep
  narrowing the gap. Slightly slower, but it never fails, so we use it as
  the dependable default.

Dividends: a stock that pays a dividend yield q drifts up a bit less. The
simplest way to handle that with our existing pricer is to feed it the
dividend-adjusted stock price S * exp(-q*T).
"""

import numpy as np
from scipy.optimize import brentq

from src.black_scholes import price as bs_price
from src.greeks import vega


def _price_bounds(S, K, T, r, option_type):
    """No-arbitrage range a European option price must sit inside."""
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        return max(S - disc_K, 0.0), S
    return max(disc_K - S, 0.0), disc_K


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0):
    """
    Implied volatility using Brent's method.
    Returns np.nan when no volatility can explain the price (e.g. the price is
    below the option's intrinsic value, which happens with stale quotes).
    """
    S_adj = S * np.exp(-q * T)
    low, high = _price_bounds(S_adj, K, T, r, option_type)
    if not (low + 1e-8 < market_price < high) or T <= 0:
        return np.nan

    def gap(sigma):
        return bs_price(S_adj, K, T, r, sigma, option_type) - market_price

    try:
        return brentq(gap, 1e-4, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan


def newton_implied_vol(market_price, S, K, T, r, option_type="call", q=0.0,
                       guess=0.5, tol=1e-8, max_iter=50):
    """
    Newton's method, kept for learning: returns (sigma, list_of_guesses) so you
    can watch it converge. new_guess = guess - (model - market) / vega
    """
    S_adj = S * np.exp(-q * T)
    sigma, history = guess, [guess]
    for _ in range(max_iter):
        diff = bs_price(S_adj, K, T, r, sigma, option_type) - market_price
        slope = vega(S_adj, K, T, r, sigma) * 100   # vega was per 1 vol point
        if abs(diff) < tol or slope < 1e-12:
            break
        sigma = sigma - diff / slope
        history.append(sigma)
    return sigma, history
