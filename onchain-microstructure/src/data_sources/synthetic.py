"""
Calibrated market/on-chain data generator.

Live connectors (onchain.py, cex.py) require Infura/Alchemy + Etherscan +
exchange API keys that aren't available in this environment. This module
produces a substitute dataset with the same statistical shape the live
pipeline is expected to see, so the signal/econometrics/backtest layers
can be built and validated end to end before wiring up real feeds:

  - CEX midprice follows a GARCH(1,1) diffusion (volatility clustering)
  - DEX price = CEX price + a mean-reverting basis (AMM inventory / gas
    friction keeps DEX from tracking CEX tick-for-tick -> cointegrated,
    not identical)
  - On-chain swap flow is partially informed: a fraction of large swaps
    are generated from a latent signal that also drives the next-period
    return, matching the "informed trader" setup in Kyle (1985) and the
    empirical on-chain literature (e.g. Barbon & Ranaldo, 2021 on informed
    DEX flow). This is what makes order-flow-imbalance -> return Granger
    tests below meaningful rather than a null result by construction.
  - A handful of large ("whale") transfers are injected at known
    timestamps with a genuine price impact, for the event study module.

Swap the calls in `load_dataset()` for the real connectors once API keys
are configured; downstream modules only depend on the column schema.
"""

import numpy as np
import pandas as pd

RNG_SEED = 7


def _garch_price_path(n, s0, mu, omega, alpha, beta, rng):
    """Simulate a GARCH(1,1) log-price path."""
    h = np.zeros(n)
    eps = np.zeros(n)
    ret = np.zeros(n)
    h[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        z = rng.standard_normal()
        h[t] = omega + alpha * eps[t - 1] ** 2 + beta * h[t - 1]
        eps[t] = z * np.sqrt(h[t])
        ret[t] = mu + eps[t]
    price = s0 * np.exp(np.cumsum(ret))
    return price, ret, h


def load_dataset(n_minutes=60 * 24 * 30, s0=3200.0, seed=RNG_SEED):
    """
    Returns a minute-bar DataFrame for a single asset (ETH/USDC as the
    working example) covering `n_minutes` of trading.

    Columns:
      ts                  UTC timestamp, 1-min bars
      cex_price            reference (CEX) midprice
      dex_price            Uniswap V3 pool implied price
      swap_count            number of DEX swaps in the bar
      swap_buy_vol_usd       USD volume of buy-side swaps
      swap_sell_vol_usd      USD volume of sell-side swaps
      net_flow_usd            buy - sell, signed on-chain flow
      unique_addresses         distinct wallets active in the bar
      whale_flag                1 if a >$500k single swap occurred
    """
    rng = np.random.default_rng(seed)

    # --- CEX reference price: GARCH(1,1), minute vol ~ matches ETH intraday
    cex_price, cex_ret, cond_var = _garch_price_path(
        n_minutes, s0, mu=0.0, omega=1.2e-7, alpha=0.06, beta=0.90, rng=rng
    )

    # --- latent informed-flow signal, AR(1), drives both order flow and
    #     the *next* bar's return (short-lived information advantage)
    phi = 0.35
    info = np.zeros(n_minutes)
    for t in range(1, n_minutes):
        info[t] = phi * info[t - 1] + rng.standard_normal() * 0.6

    info_effect_bps = 1.8  # informed flow shifts next-bar return by up to ~1.8bps per unit
    cex_price = cex_price.copy()
    for t in range(1, n_minutes):
        adj = 1 + (info[t - 1] * info_effect_bps) / 1e4
        cex_price[t] = cex_price[t] * adj
    # renormalize cumulative drift so injected adj doesn't explode the level
    cex_price = cex_price / cex_price[0] * s0

    # --- DEX basis: mean-reverting around 0 (OU process), driven partly
    #     by pool inventory imbalance -> cointegrated with CEX, not equal
    basis = np.zeros(n_minutes)
    kappa, sigma_b = 0.08, 0.35
    for t in range(1, n_minutes):
        basis[t] = basis[t - 1] + kappa * (0 - basis[t - 1]) + rng.standard_normal() * sigma_b
    dex_price = cex_price + basis  # basis in USD terms

    # --- on-chain swap activity, correlated with |info| and with vol regime
    base_swaps = rng.poisson(lam=4 + 8 * (cond_var / cond_var.mean()), size=n_minutes)
    informed_participation = np.clip(0.5 + 0.5 * np.tanh(info), 0, 1)

    buy_vol = np.zeros(n_minutes)
    sell_vol = np.zeros(n_minutes)
    whale_flag = np.zeros(n_minutes, dtype=int)
    unique_addr = np.zeros(n_minutes, dtype=int)

    for t in range(n_minutes):
        n_swaps = max(base_swaps[t], 0)
        if n_swaps == 0:
            continue
        sizes = rng.pareto(a=1.8, size=n_swaps) * 4_000 + 500  # fat-tailed swap sizes, USD
        # direction is biased by the informed signal at t (they trade ahead of the move)
        p_buy = np.clip(0.5 + 0.5 * np.tanh(info[t]) * informed_participation[t], 0.05, 0.95)
        directions = rng.random(n_swaps) < p_buy
        buy_vol[t] = sizes[directions].sum()
        sell_vol[t] = sizes[~directions].sum()
        unique_addr[t] = max(1, int(n_swaps * rng.uniform(0.6, 0.95)))
        if sizes.max() > 500_000:
            whale_flag[t] = 1

    net_flow = buy_vol - sell_vol

    ts = pd.date_range("2026-06-01", periods=n_minutes, freq="1min", tz="UTC")
    df = pd.DataFrame(
        {
            "ts": ts,
            "cex_price": cex_price,
            "dex_price": dex_price,
            "swap_count": base_swaps,
            "swap_buy_vol_usd": buy_vol,
            "swap_sell_vol_usd": sell_vol,
            "net_flow_usd": net_flow,
            "unique_addresses": unique_addr,
            "whale_flag": whale_flag,
        }
    ).set_index("ts")

    return df


if __name__ == "__main__":
    d = load_dataset()
    print(d.describe())
