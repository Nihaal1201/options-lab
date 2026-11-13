"""
Monte Carlo option pricing.

Intuition
---------
Imagine running the future 100,000 times. In each 'universe' the stock
takes a random walk, and you record what the option pays at expiry.
The option's fair price = the AVERAGE payoff across all universes,
discounted back to today.

How the stock moves (Geometric Brownian Motion, the Black-Scholes model):
    S_T = S * exp( (r - 0.5*sigma^2) * T  +  sigma * sqrt(T) * Z )
where Z is a random draw from a standard normal distribution (the 'dice roll').

Why bother when Black-Scholes exists? Because Monte Carlo works for options
that have NO formula, e.g. Asian options (payoff depends on the AVERAGE
price over the life of the option). That's how real desks price exotics.

Every function returns (price, standard_error). The standard error tells you
how much the estimate would wobble if you re-ran it with new random numbers.
A 95% confidence interval is roughly price +/- 1.96 * standard_error.
"""

import numpy as np


def simulate_terminal_prices(S, T, r, sigma, n_paths, rng, antithetic=False):
    """Stock prices at expiry for n_paths random universes."""
    if antithetic:
        # Variance reduction: for every dice roll Z, also use -Z.
        # The two mirror-image paths cancel out a lot of noise.
        half = rng.standard_normal(n_paths // 2)
        Z = np.concatenate([half, -half])
    else:
        Z = rng.standard_normal(n_paths)
    return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def simulate_paths(S, T, r, sigma, n_paths, n_steps, rng):
    """Full price paths (n_paths rows x n_steps+1 columns), step by step."""
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_steps = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(log_steps, axis=1)
    paths = S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return paths


def _discounted_mean(payoffs, r, T):
    discounted = np.exp(-r * T) * payoffs
    price = discounted.mean()
    std_error = discounted.std(ddof=1) / np.sqrt(len(discounted))
    return price, std_error


def mc_european(S, K, T, r, sigma, n_paths=100_000, option_type="call",
                antithetic=False, seed=None):
    """Monte Carlo price of a European option. Returns (price, std_error)."""
    rng = np.random.default_rng(seed)
    ST = simulate_terminal_prices(S, T, r, sigma, n_paths, rng, antithetic)
    payoffs = np.maximum(ST - K, 0) if option_type == "call" else np.maximum(K - ST, 0)
    if antithetic:
        # Average each pair first so the standard error is measured honestly
        half = n_paths // 2
        payoffs = 0.5 * (payoffs[:half] + payoffs[half:])
    return _discounted_mean(payoffs, r, T)


def mc_asian(S, K, T, r, sigma, n_paths=50_000, n_steps=252,
             option_type="call", seed=None):
    """
    Arithmetic-average Asian option: payoff uses the AVERAGE price over the
    option's life instead of the final price. No closed-form formula exists.
    Returns (price, std_error).
    """
    rng = np.random.default_rng(seed)
    paths = simulate_paths(S, T, r, sigma, n_paths, n_steps, rng)
    avg_price = paths[:, 1:].mean(axis=1)
    payoffs = np.maximum(avg_price - K, 0) if option_type == "call" else np.maximum(K - avg_price, 0)
    return _discounted_mean(payoffs, r, T)
