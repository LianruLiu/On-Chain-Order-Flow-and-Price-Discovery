"""
Standard event-study machinery (MacKinlay, 1997) applied to on-chain
whale transactions: is there measurable price impact / drift around
large swaps, and does it persist (informed flow) or revert (liquidity
shock that gets arbed away)?
"""

import numpy as np
import pandas as pd


def find_events(df: pd.DataFrame) -> pd.DatetimeIndex:
    return df.index[df["whale_flag"] == 1]


def car_study(price: pd.Series, events: pd.DatetimeIndex, pre: int = 30, post: int = 60) -> pd.DataFrame:
    """
    For each event timestamp, build the abnormal-return path in
    [-pre, +post] minutes around it. "Abnormal" here is the raw log
    return net of the sample's average minute return (a market-model
    with a single constant expected return, the simplest MacKinlay
    specification — fine for a same-asset, high-frequency window).
    """
    ret = np.log(price).diff()
    mean_ret = ret.mean()
    abnormal = ret - mean_ret

    idx = price.index
    paths = []
    for ev in events:
        pos = idx.get_indexer([ev])[0]
        if pos - pre < 0 or pos + post >= len(idx):
            continue
        window = abnormal.iloc[pos - pre : pos + post + 1].values
        paths.append(window)

    if not paths:
        return pd.DataFrame()

    arr = np.array(paths)
    car = np.cumsum(arr, axis=1)
    offsets = np.arange(-pre, post + 1)

    result = pd.DataFrame(car.T, index=offsets)
    result["mean_car"] = result.mean(axis=1)
    result["se"] = result.iloc[:, :-1].std(axis=1) / np.sqrt(len(paths))
    return result[["mean_car", "se"]]
