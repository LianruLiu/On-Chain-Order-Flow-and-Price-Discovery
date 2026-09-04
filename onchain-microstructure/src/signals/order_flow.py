"""
Order-flow signals built from on-chain swap volume.

ofi(): normalized order flow imbalance, (buy - sell) / (buy + sell),
smoothed over a rolling window. Standard order-book microstructure
measure (Cont, Kukanov & Stoikov, 2014), adapted here to signed DEX
swap volume in place of book-side quote updates.

vpin(): volume-synchronized probability of informed trading (Easley,
Lopez de Prado & O'Hara, 2012). Buckets trades by fixed volume rather
than clock time, then measures |buy - sell| / total volume per bucket
as a proxy for flow toxicity.
"""

import numpy as np
import pandas as pd


def ofi(df: pd.DataFrame, window: int = 15) -> pd.Series:
    buy = df["swap_buy_vol_usd"]
    sell = df["swap_sell_vol_usd"]
    total = (buy + sell).clip(lower=1.0)
    raw = (buy - sell) / total
    return raw.rolling(window, min_periods=1).mean().rename("ofi")


def vpin(df: pd.DataFrame, bucket_usd: float = 250_000, n_buckets: int = 50) -> pd.Series:
    vol = (df["swap_buy_vol_usd"] + df["swap_sell_vol_usd"]).values
    imb = (df["swap_buy_vol_usd"] - df["swap_sell_vol_usd"]).abs().values
    cum_vol = np.cumsum(vol)

    n_full_buckets = int(cum_vol[-1] // bucket_usd) if cum_vol[-1] > 0 else 0
    bucket_edges = bucket_usd * np.arange(1, n_full_buckets + 1)
    bucket_idx = np.searchsorted(cum_vol, bucket_edges)

    out = pd.Series(index=df.index, dtype=float)
    prev = 0
    vpin_vals = []
    for edge_pos in bucket_idx:
        edge_pos = min(edge_pos, len(df) - 1)
        b_vol = vol[prev : edge_pos + 1].sum()
        b_imb = imb[prev : edge_pos + 1].sum()
        vpin_vals.append(b_imb / b_vol if b_vol > 0 else np.nan)
        out.iloc[edge_pos] = vpin_vals[-1]
        prev = edge_pos + 1

    return out.rolling(n_buckets, min_periods=1).mean().ffill().rename("vpin")


def net_flow_zscore(df: pd.DataFrame, window: int = 60) -> pd.Series:
    nf = df["net_flow_usd"]
    mu = nf.rolling(window, min_periods=10).mean()
    sd = nf.rolling(window, min_periods=10).std().clip(lower=1.0)
    return ((nf - mu) / sd).rename("net_flow_z")
