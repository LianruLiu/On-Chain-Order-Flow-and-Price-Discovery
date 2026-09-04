"""
GARCH(1,1) on CEX returns, then a variant with on-chain swap count as
an exogenous variance regressor (GARCH-X) to test whether observable
on-chain activity explains volatility clustering beyond past returns
alone.
"""

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(returns: pd.Series):
    r = returns.dropna() * 100  # arch expects returns in %
    am = arch_model(r, vol="Garch", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    return res


def fit_garch_x(returns: pd.Series, exog: pd.Series):
    """
    arch's GARCH-X isn't exposed directly for the variance equation in
    a one-liner, so this fits GARCH(1,1) on returns and separately
    regresses squared residuals on the exogenous activity series to
    quantify explanatory power (a standard two-step approximation to
    a full GARCH-X MLE).
    """
    base = fit_garch(returns)
    resid_sq = base.resid**2
    x = pd.concat([resid_sq.rename("resid_sq"), exog.reindex(resid_sq.index)], axis=1).dropna()

    import statsmodels.api as sm

    model = sm.OLS(x["resid_sq"], sm.add_constant(x.iloc[:, 1])).fit()
    return {
        "garch_result": base,
        "exog_coef": model.params.iloc[1],
        "exog_pvalue": model.pvalues.iloc[1],
        "exog_r2": model.rsquared,
    }
