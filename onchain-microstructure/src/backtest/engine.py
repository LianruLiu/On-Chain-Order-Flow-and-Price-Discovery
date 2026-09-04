"""
Minimal vectorized backtest: turn a signal into a position, apply a
cost model, report the stats an allocator actually asks for.

Position sizing is a plain threshold rule (long above +z, short below
-z, flat otherwise) — the point of this repo is signal validity, not
a tuned execution strategy, so the backtest is kept deliberately
simple rather than dressed up with parameters that were fit in-sample.
"""

import numpy as np
import pandas as pd


def run_backtest(
    price: pd.Series,
    signal: pd.Series,
    entry_z: float = 1.0,
    exit_z: float = 0.25,
    cost_bps: float = 5.0,
    hold_bars: int = 5,
) -> dict:
    ret = price.pct_change().shift(-1)  # next-bar return, avoid lookahead
    sig = signal.reindex(price.index)

    pos = np.zeros(len(price))
    state = 0
    bars_in_state = 0
    for i in range(len(price)):
        s = sig.iloc[i]
        if np.isnan(s):
            pos[i] = state
            continue
        if state == 0:
            if s > entry_z:
                state, bars_in_state = 1, 0
            elif s < -entry_z:
                state, bars_in_state = -1, 0
        else:
            bars_in_state += 1
            # minimum holding period before an exit is allowed, to keep
            # turnover (and cost drag) from swamping a noisy minute-bar signal
            if bars_in_state >= hold_bars and abs(s) < exit_z:
                state, bars_in_state = 0, 0
        pos[i] = state

    pos = pd.Series(pos, index=price.index)
    trades = pos.diff().abs().fillna(0)
    cost = trades * (cost_bps / 1e4)

    # pos[i] is decided using information available at the close of bar i;
    # ret[i] is already the forward (i -> i+1) return, so no extra shift here
    strat_ret = pos.fillna(0) * ret.fillna(0) - cost
    equity = (1 + strat_ret).cumprod()

    n_per_year = 60 * 24 * 365  # minute bars
    sharpe = (
        strat_ret.mean() / strat_ret.std() * np.sqrt(n_per_year)
        if strat_ret.std() > 0
        else np.nan
    )
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()

    return {
        "equity_curve": equity,
        "returns": strat_ret,
        "position": pos,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "total_return": equity.iloc[-1] - 1,
        "n_trades": int(trades.sum() / 2),
        "hit_rate": float((strat_ret[strat_ret != 0] > 0).mean()) if (strat_ret != 0).any() else np.nan,
    }
