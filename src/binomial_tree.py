"""
Cox-Ross-Rubinstein binomial tree option pricing.

Breaks time to expiry into n discrete steps. At each step, the stock can
move up by a factor u or down by a factor d. Working backward from expiry
(where the option's value is just its payoff) to today, folding in the
risk-neutral probability of each move, gives the option's price today.

As n -> infinity, this converges to the Black-Scholes price -- this
discrete random walk becomes the continuous one Black-Scholes assumes.

This method can also price American options (early exercise allowed at
any step), which the closed-form Black-Scholes formula cannot do.
"""

import numpy as np


def binomial_tree_price(S, K, T, r, sigma, n_steps=500, option_type="call",
                         exercise_style="european"):
    """
    Price an option via a CRR binomial tree.

    exercise_style: "european" (only exercisable at expiry) or
                     "american" (exercisable at any step)
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))       # up-move factor
    d = 1 / u                              # down-move factor, CRR convention
    p = (np.exp(r * dt) - d) / (u - d)    # risk-neutral probability of an up-move
    discount = np.exp(-r * dt)

    # Stock prices at expiry (final layer of the tree): after n_steps,
    # there are n_steps+1 possible outcomes, from n_steps down-moves to
    # n_steps up-moves.
    j = np.arange(n_steps + 1)
    stock_at_expiry = S * (u ** j) * (d ** (n_steps - j))

    if option_type == "call":
        values = np.maximum(stock_at_expiry - K, 0)
    elif option_type == "put":
        values = np.maximum(K - stock_at_expiry, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Work backward from expiry to today, one step at a time.
    for step in range(n_steps - 1, -1, -1):
        # Expected value under risk-neutral probabilities, discounted one step
        values = discount * (p * values[1:] + (1 - p) * values[:-1])

        if exercise_style == "american":
            # At each step, the holder could instead exercise immediately --
            # so the option is worth at least its immediate exercise value.
            j = np.arange(step + 1)
            stock_at_step = S * (u ** j) * (d ** (step - j))
            if option_type == "call":
                immediate_exercise = np.maximum(stock_at_step - K, 0)
            else:
                immediate_exercise = np.maximum(K - stock_at_step, 0)
            values = np.maximum(values, immediate_exercise)

    return values[0]
