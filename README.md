# On-Chain Order Flow and Price Discovery

A market microstructure study applying classical order-flow and
cointegration methods — Kyle (1985), Easley/López de Prado/O'Hara VPIN,
Engle-Granger, MacKinlay event studies — to transaction-level DEX data
instead of the limit-order-book data those methods were originally built
for.

**The project lives in [`onchain-microstructure/`](onchain-microstructure/)**
— start from its [README](onchain-microstructure/README.md) and the
[research note](onchain-microstructure/reports/research_note.md).

## Quickstart

```bash
cd onchain-microstructure
pip install -r requirements.txt
python run_analysis.py
```

This regenerates everything under `onchain-microstructure/reports/`
(figures + research note) from a calibrated synthetic dataset — no API
keys needed. See the nested README for the methodology, the "why synthetic
data" rationale, and how to point the pipeline at live Uniswap V3 / CEX
feeds (`requirements-prod.txt`).
