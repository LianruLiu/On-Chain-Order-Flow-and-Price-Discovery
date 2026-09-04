"""
Reference CEX price feed via ccxt, used as the benchmark leg for the
DEX-CEX cointegration test and as the return series for the event study.

    import ccxt
    ex = ccxt.binance()
    ohlcv = ex.fetch_ohlcv("ETH/USDT", timeframe="1m", since=..., limit=1000)

Not executed here (no outbound network to exchange APIs in this sandbox).
`synthetic.py` stands in with a GARCH-simulated series of matching shape.
"""

import os


def fetch_ohlcv(symbol: str, timeframe: str, since, limit: int = 1000):
    try:
        import ccxt
    except ImportError as e:
        raise RuntimeError("pip install ccxt") from e

    exchange_id = os.environ.get("CEX_EXCHANGE", "binance")
    exchange = getattr(ccxt, exchange_id)()
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
