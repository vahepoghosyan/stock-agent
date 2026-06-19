"""Generate Japanese candlestick charts for portfolio stocks."""

from __future__ import annotations

import io
from typing import Optional

import mplfinance as mpf
import numpy as np
import yfinance as yf


_STYLE = mpf.make_mpf_style(
    base_mpf_style="charles",
    marketcolors=mpf.make_marketcolors(
        up="#16a34a", down="#dc2626",
        wick={"up": "#16a34a", "down": "#dc2626"},
        volume={"up": "#bbf7d0", "down": "#fecaca"},
        edge={"up": "#16a34a", "down": "#dc2626"},
    ),
    gridstyle="--",
    gridcolor="#e5e7eb",
    facecolor="#ffffff",
    figcolor="#ffffff",
    rc={"font.size": 9, "axes.labelcolor": "#374151", "xtick.color": "#6b7280", "ytick.color": "#6b7280"},
)


def generate_candlestick_chart(ticker: str, period: str = "3mo") -> Optional[bytes]:
    """Return PNG bytes for a 3-month candlestick chart, or None on failure."""
    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if data.empty:
            return None
        # yfinance may return multi-level columns; flatten to single level
        if isinstance(data.columns, __import__("pandas").MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.tail(60)

        buf = io.BytesIO()
        mpf.plot(
            data,
            type="candle",
            style=_STYLE,
            title=f"  {ticker} — 3 Month",
            ylabel="Price (USD)",
            volume=True,
            figsize=(8, 4.5),
            tight_layout=True,
            savefig=dict(fname=buf, dpi=110, bbox_inches="tight"),
        )
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


def generate_1h_candlestick_chart(ticker: str) -> Optional[bytes]:
    """Return PNG bytes for a 1-hour interval candlestick chart (last 5 days), or None on failure."""
    try:
        data = yf.download(ticker, period="5d", interval="1h", progress=False, auto_adjust=True)
        if data.empty:
            return None
        if isinstance(data.columns, __import__("pandas").MultiIndex):
            data.columns = data.columns.get_level_values(0)

        close = data["Close"].values.astype(float)
        high  = data["High"].values.astype(float)
        low   = data["Low"].values.astype(float)
        x = np.arange(len(close), dtype=float)

        uptrend = np.polyfit(x, close, 1)[0] >= 0
        prices  = low if uptrend else high

        # Find tightest line through two candle extremes that never crosses while trend holds.
        # Uptrend: highest line still at-or-below every low (support).
        # Downtrend: lowest line still at-or-above every high (resistance).
        # When trend reverses, the line naturally crosses the newer candles.
        best: tuple[float, float] | None = None
        n = len(prices)
        for i in range(n - 1):
            for j in range(i + 1, n):
                s = (prices[j] - prices[i]) / (j - i)
                b = prices[i] - s * i
                line = s * x + b
                if uptrend:
                    if np.all(low >= line - 1e-8):
                        if best is None or s * x[-1] + b > best[0] * x[-1] + best[1]:
                            best = (s, b)
                else:
                    if np.all(high <= line + 1e-8):
                        if best is None or s * x[-1] + b < best[0] * x[-1] + best[1]:
                            best = (s, b)

        if best is None:
            s, b = np.polyfit(x, close, 1)
            best = (s, b)

        trendline = best[0] * x + best[1]
        ap = mpf.make_addplot(trendline, color="#f59e0b", linewidths=1.5, linestyle="--")

        buf = io.BytesIO()
        mpf.plot(
            data,
            type="candle",
            style=_STYLE,
            title=f"  {ticker} — 1 Hour",
            ylabel="Price (USD)",
            volume=True,
            addplot=ap,
            figsize=(8, 4.5),
            tight_layout=True,
            savefig=dict(fname=buf, dpi=110, bbox_inches="tight"),
        )
        buf.seek(0)
        return buf.read()
    except Exception:
        return None
