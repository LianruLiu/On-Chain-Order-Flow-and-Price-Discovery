"""
Does on-chain order flow lead price, or just react to it?

Tests both directions (flow -> returns, returns -> flow) at several
lags. Reports the lag-by-lag p-values plus, for the flow -> returns
direction, the incremental R^2 from adding flow to an AR(k) return
model — the p-value alone doesn't tell you if the effect is tradeable.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


def granger_matrix(returns: pd.Series, flow: pd.Series, max_lag: int = 10) -> pd.DataFrame:
    data = pd.concat([returns.rename("ret"), flow.rename("flow")], axis=1).dropna()

    rows = []
    for direction, cols in [("flow_to_returns", ["ret", "flow"]), ("returns_to_flow", ["flow", "ret"])]:
        res = grangercausalitytests(data[cols].values, maxlag=max_lag)
        for lag, stats in res.items():
            f_stat, p_value = stats[0]["ssr_ftest"][0], stats[0]["ssr_ftest"][1]
            rows.append({"direction": direction, "lag": lag, "f_stat": f_stat, "p_value": p_value})
    return pd.DataFrame(rows)


def incremental_r2(returns: pd.Series, flow: pd.Series, lag: int = 1, ar_order: int = 3) -> dict:
    """R^2 of AR(ar_order) return model, vs. same model + lagged flow."""
    df = pd.DataFrame({"ret": returns})
    for k in range(1, ar_order + 1):
        df[f"ret_l{k}"] = df["ret"].shift(k)
    df["flow_l"] = flow.shift(lag)
    df = df.dropna()

    ar_cols = [f"ret_l{k}" for k in range(1, ar_order + 1)]
    base = sm.OLS(df["ret"], sm.add_constant(df[ar_cols])).fit()
    full = sm.OLS(df["ret"], sm.add_constant(df[ar_cols + ["flow_l"]])).fit()

    return {
        "lag": lag,
        "r2_base": base.rsquared,
        "r2_with_flow": full.rsquared,
        "delta_r2": full.rsquared - base.rsquared,
        "flow_coef": full.params["flow_l"],
        "flow_pvalue": full.pvalues["flow_l"],
    }
