"""
DEX-CEX cointegration and the implied statistical-arbitrage spread.

engle_granger(): two-step EG test — regress dex on cex, test residual
stationarity with ADF. Reports the hedge ratio and half-life of mean
reversion (via OU fit on the residual), which is the number a
stat-arb desk actually trades off.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint


def engle_granger(dex: pd.Series, cex: pd.Series) -> dict:
    x = sm.add_constant(cex.values)
    model = sm.OLS(dex.values, x).fit()
    hedge_ratio = model.params[1]
    intercept = model.params[0]
    resid = dex.values - (intercept + hedge_ratio * cex.values)

    adf_stat, adf_p, *_ = adfuller(resid, autolag="AIC")
    coint_stat, coint_p, crit = coint(dex.values, cex.values)

    half_life = _ou_half_life(resid)

    return {
        "hedge_ratio": float(hedge_ratio),
        "intercept": float(intercept),
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_p),
        "engle_granger_stat": float(coint_stat),
        "engle_granger_pvalue": float(coint_p),
        "half_life_minutes": half_life,
        "residual": pd.Series(resid, index=dex.index, name="spread"),
    }


def _ou_half_life(resid: np.ndarray) -> float:
    """Fit spread_t - spread_{t-1} = theta * spread_{t-1} + eps, half-life = -ln(2)/theta."""
    y = resid[1:] - resid[:-1]
    x = sm.add_constant(resid[:-1])
    fit = sm.OLS(y, x).fit()
    theta = fit.params[1]
    if theta >= 0:
        return float("inf")
    return float(-np.log(2) / theta)
