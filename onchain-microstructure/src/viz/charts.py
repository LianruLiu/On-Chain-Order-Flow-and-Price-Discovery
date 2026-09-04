import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from . import style

style.apply()


def price_flow_panel(df, ofi_series, path, window=("2026-06-05 00:00", "2026-06-05 08:00")):
    d = df.loc[window[0]:window[1]]
    ofi_s = ofi_series.loc[window[0]:window[1]]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True, height_ratios=[2.2, 1], gridspec_kw={"hspace": 0.08}
    )

    ax1.plot(d.index, d["cex_price"], color=style.ACCENT_1, lw=1.1, label="CEX (reference)")
    ax1.plot(d.index, d["dex_price"], color=style.ACCENT_2, lw=0.9, alpha=0.85, label="DEX (Uniswap V3)")
    ax1.set_title("ETH/USDC — CEX vs. DEX price, with on-chain order-flow imbalance")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_ylabel("USD")

    ax2.fill_between(
        ofi_s.index, 0, ofi_s.values,
        where=ofi_s.values >= 0, color=style.ACCENT_BUY, alpha=0.6, linewidth=0,
    )
    ax2.fill_between(
        ofi_s.index, 0, ofi_s.values,
        where=ofi_s.values < 0, color=style.ACCENT_SELL, alpha=0.6, linewidth=0,
    )
    ax2.axhline(0, color=style.MUTED, lw=0.6)
    ax2.set_ylabel("OFI (15m)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2.set_xlabel("UTC")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cointegration_spread(residual, eg_result, path):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(residual.index, residual.values, color=style.ACCENT_1, lw=0.8)
    mu = residual.mean()
    sd = residual.std()
    ax.axhline(mu, color=style.MUTED, lw=0.8, ls="--")
    ax.axhline(mu + sd, color=style.ACCENT_2, lw=0.6, ls=":")
    ax.axhline(mu - sd, color=style.ACCENT_2, lw=0.6, ls=":")
    ax.set_title(
        f"DEX–CEX spread (Engle–Granger residual)  "
        f"|  ADF p={eg_result['adf_pvalue']:.4f}, half-life≈{eg_result['half_life_minutes']:.0f}m"
    )
    ax.set_ylabel("USD")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def granger_heatmap(gdf, path):
    piv = gdf.pivot(index="direction", columns="lag", values="p_value")
    fig, ax = plt.subplots(figsize=(9, 2.4))
    im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.15)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(["flow → returns", "returns → flow"])
    ax.set_xlabel("lag (minutes)")
    ax.set_title("Granger causality p-values by lag (green = significant)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7, color="#0e1117" if v < 0.08 else "#c9d1d9")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def event_study_car(car_df, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    x = car_df.index.values
    y = car_df["mean_car"].values * 1e4  # bps
    se = car_df["se"].values * 1e4
    ax.plot(x, y, color=style.ACCENT_1, lw=1.4)
    ax.fill_between(x, y - 1.96 * se, y + 1.96 * se, color=style.ACCENT_1, alpha=0.18, linewidth=0)
    ax.axvline(0, color=style.ACCENT_SELL, lw=0.8, ls="--")
    ax.axhline(0, color=style.MUTED, lw=0.6)
    ax.set_title("CAR around whale swaps (>$500k), event window [-30, +60] min")
    ax.set_xlabel("minutes relative to event")
    ax.set_ylabel("cumulative abnormal return (bps)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def backtest_summary(bt, path):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5.5), sharex=True, height_ratios=[2, 1], gridspec_kw={"hspace": 0.08}
    )
    eq = bt["equity_curve"]
    ax1.plot(eq.index, eq.values, color=style.ACCENT_1, lw=1.2)
    ax1.set_title(
        f"OFI-threshold strategy — Sharpe {bt['sharpe']:.2f}, "
        f"max DD {bt['max_drawdown']*100:.1f}%, {bt['n_trades']} trades"
    )
    ax1.set_ylabel("equity (norm.)")

    dd = eq / eq.cummax() - 1
    ax2.fill_between(dd.index, 0, dd.values * 100, color=style.ACCENT_SELL, alpha=0.5, linewidth=0)
    ax2.set_ylabel("drawdown (%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def garch_vol(res, path):
    fig, ax = plt.subplots(figsize=(11, 3.2))
    cv = res.conditional_volatility
    ax.plot(cv.index, cv.values, color=style.ACCENT_2, lw=0.9)
    ax.set_title("GARCH(1,1)-t conditional volatility, 1-min ETH/USDC returns")
    ax.set_ylabel("σ (%)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
