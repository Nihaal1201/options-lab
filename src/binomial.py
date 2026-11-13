"""
Binomial tree pricing (Cox-Ross-Rubinstein model).

Intuition
---------
Instead of a smooth formula, chop the time to expiry into N small steps.
At each step the stock can only do one of two things: go UP by a factor u
or DOWN by a factor d. That builds a tree of possible future prices.

1. At expiry (the last column of the tree), the option is worth its payoff.
2. Walk BACKWARDS one step at a time. Each node is worth the
   probability-weighted average of its two children, discounted one step.
3. For an AMERICAN option, at every node also ask: "is exercising right now
   worth more than holding on?" Keep whichever is bigger.

As N grows, the tree's price converges to Black-Scholes (for European
options). Its real power: it can price American options, which
Black-Scholes cannot.

Inputs are the same as black_scholes.py, plus:
    N        : number of time steps in the tree
    american : True lets you exercise early
"""

import numpy as np


def binomial_price(S, K, T, r, sigma, N=200, option_type="call", american=False):
    dt = T / N                              # length of one step (in years)
    u = np.exp(sigma * np.sqrt(dt))         # up factor
    d = 1 / u                               # down factor
    p = (np.exp(r * dt) - d) / (u - d)      # risk-neutral probability of 'up'
    disc = np.exp(-r * dt)                  # discount factor for one step

    # Stock prices at expiry: j = number of up-moves (0..N)
    j = np.arange(N + 1)
    stock = S * u**j * d**(N - j)

    # Option value at expiry = payoff
    if option_type == "call":
        values = np.maximum(stock - K, 0)
    else:
        values = np.maximum(K - stock, 0)

    # Walk backwards through the tree
    for step in range(N - 1, -1, -1):
        values = disc * (p * values[1:] + (1 - p) * values[:-1])
        if american:
            j = np.arange(step + 1)
            stock = S * u**j * d**(step - j)
            exercise_now = stock - K if option_type == "call" else K - stock
            values = np.maximum(values, exercise_now)

    return values[0]
