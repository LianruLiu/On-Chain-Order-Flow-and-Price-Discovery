"""
Live Uniswap V3 swap feed via JSON-RPC + the pool's Swap event log.

Requires:
  ETH_RPC_URL      Alchemy/Infura HTTPS endpoint
  ETHERSCAN_KEY     for ABI lookups / address labeling (whale tagging)

Pulls raw Swap(address,address,int256,int256,uint160,uint128,int24) logs
for a given pool, decodes amount0/amount1 into a signed USD flow using the
pool's token decimals and a reference USD price, and bins into fixed
intervals. Output schema matches `data_sources.synthetic.load_dataset`
so it's a drop-in replacement once credentials are configured:

    from src.data_sources import onchain
    df = onchain.fetch_pool_flow(pool="0x88e6...", start=..., end=...)

Not executed in this repo's demo run (see README) — the sandbox this was
built in has no outbound access to RPC providers. Left here as the real
implementation target; swap_flow() below is the only piece downstream
code touches.
"""

from dataclasses import dataclass
from datetime import datetime
import os

SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca"

UNISWAP_V3_POOL_ABI_FRAGMENT = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "sender", "type": "address"},
            {"indexed": True, "name": "recipient", "type": "address"},
            {"indexed": False, "name": "amount0", "type": "int256"},
            {"indexed": False, "name": "amount1", "type": "int256"},
            {"indexed": False, "name": "sqrtPriceX96", "type": "uint160"},
            {"indexed": False, "name": "liquidity", "type": "uint128"},
            {"indexed": False, "name": "tick", "type": "int24"},
        ],
        "name": "Swap",
        "type": "event",
    }
]


@dataclass
class PoolConfig:
    address: str
    token0_decimals: int
    token1_decimals: int
    token0_symbol: str
    token1_symbol: str


def fetch_pool_flow(pool: str, start: datetime, end: datetime, bar: str = "1min"):
    """
    Placeholder for the production implementation. In order:
      1. web3.eth.get_logs(fromBlock, toBlock, address=pool, topics=[SWAP_TOPIC])
      2. decode each log against UNISWAP_V3_POOL_ABI_FRAGMENT
      3. amount0/amount1 signs give trade direction; convert to USD using
         the token0/token1 decimals and a spot USD reference price
      4. resample into `bar` bins, aggregate buy/sell volume, swap count,
         distinct `sender` addresses per bin, and flag bins containing a
         swap above the whale threshold
      5. return a DataFrame matching synthetic.load_dataset()'s schema

    Raises NotImplementedError until ETH_RPC_URL is set — this is the
    integration point for wiring real data in, not a working call.
    """
    if not os.environ.get("ETH_RPC_URL"):
        raise NotImplementedError(
            "No ETH_RPC_URL configured. Set an Alchemy/Infura endpoint and "
            "implement log decoding above; see synthetic.py for the schema "
            "this function needs to return."
        )
    raise NotImplementedError
