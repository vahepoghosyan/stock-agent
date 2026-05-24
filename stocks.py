"""Fetch historical stock data and compute technical indicators."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


@dataclass
class TechnicalAnalysis:
    ticker: str
    current_price: float
    price_change_1d: float
    price_change_1d_pct: float
    rsi14: float
    macd_histogram: float
    macd_signal: str          # bullish_crossover | bearish_crossover | bullish | bearish | neutral
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: str          # above_upper | near_upper | middle | near_lower | below_lower
    sma50: float
    sma200: float
    ma_signal: str            # golden_cross | death_cross | above_both | below_both | neutral
    overall_signal: str       # strong_buy | buy | hold | sell | strong_sell
    score: int = field(repr=False, default=0)
    shares: int = field(repr=False, default=0)
    buy_price: float = field(repr=False, default=0.0)
    unrealized_pnl: float = field(repr=False, default=0.0)
    unrealized_pnl_pct: float = field(repr=False, default=0.0)


# ── Fee utilities ────────────────────────────────────────────────────────────

def transaction_fee(shares: int, price: float) -> float:
    """One-way brokerage fee: shares × (0.12% of price + $0.012) + $1.20 flat."""
    return shares * (price * 0.0012 + 0.012) + 1.20


def round_trip_fee(shares: int, price: float) -> float:
    """Total cost of a buy-then-sell for the given position."""
    return 2 * transaction_fee(shares, price)


def break_even_pct(shares: int, price: float) -> float:
    """Minimum % price gain needed to profit after a full round trip."""
    if shares <= 0:
        return 0.0
    return round_trip_fee(shares, price) / (shares * price) * 100


# ── Indicator calculations ──────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return float(histogram.iloc[-1]), float(histogram.iloc[-2])


def _bbands(close: pd.Series, period: int = 20, std: float = 2.0):
    sma = close.rolling(period).mean()
    sigma = close.rolling(period).std()
    upper  = sma + std * sigma
    lower  = sma - std * sigma
    return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])


def _sma(close: pd.Series, period: int):
    s = close.rolling(period).mean()
    return float(s.iloc[-1]), float(s.iloc[-2])


# ── Signal helpers ──────────────────────────────────────────────────────────

def _macd_signal_label(hist_now: float, hist_prev: float) -> str:
    if hist_prev < 0 < hist_now:      return "bullish_crossover"
    if hist_prev > 0 > hist_now:      return "bearish_crossover"
    if hist_now > 0:                  return "bullish"
    if hist_now < 0:                  return "bearish"
    return "neutral"


def _bb_position(price: float, upper: float, lower: float) -> str:
    r = upper - lower
    if price > upper:                    return "above_upper"
    if price > upper - r * 0.1:         return "near_upper"
    if price < lower:                    return "below_lower"
    if price < lower + r * 0.1:         return "near_lower"
    return "middle"


def _ma_signal(price: float, s50: float, s200: float, p50: float, p200: float) -> str:
    if p50 < p200 and s50 > s200:       return "golden_cross"
    if p50 > p200 and s50 < s200:       return "death_cross"
    if price > s50 and price > s200:    return "above_both"
    if price < s50 and price < s200:    return "below_both"
    return "neutral"


def _overall(score: int) -> str:
    if score >= 4:   return "strong_buy"
    if score >= 1:   return "buy"
    if score <= -4:  return "strong_sell"
    if score <= -1:  return "sell"
    return "hold"


# ── Main function ────────────────────────────────────────────────────────────

def analyze_stock(ticker: str, shares: int = 0, buy_price: float = 0.0) -> TechnicalAnalysis:
    """Download ~210 days of daily OHLCV data and compute all indicators."""
    df = yf.Ticker(ticker).history(period="210d", interval="1d")
    if len(df) < 35:
        raise ValueError(f"Too little data for {ticker}: only {len(df)} rows")

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    price      = float(close.iloc[-1])
    prev_price = float(close.iloc[-2])
    change     = price - prev_price
    change_pct = change / prev_price * 100

    rsi14                  = _rsi(close)
    hist_now, hist_prev    = _macd(close)
    bb_upper, bb_mid, bb_low = _bbands(close)
    sma50,  prev50         = _sma(close, 50)
    sma200, prev200        = _sma(close, 200)

    macd_sig = _macd_signal_label(hist_now, hist_prev)
    bb_pos   = _bb_position(price, bb_upper, bb_low)
    ma_sig   = _ma_signal(price, sma50, sma200, prev50, prev200)

    # Composite score (-8 … +8)
    score = 0
    if rsi14 < 30:     score += 2
    elif rsi14 < 40:   score += 1
    elif rsi14 > 70:   score -= 2
    elif rsi14 > 60:   score -= 1

    if macd_sig == "bullish_crossover":   score += 2
    elif macd_sig == "bullish":           score += 1
    elif macd_sig == "bearish_crossover": score -= 2
    elif macd_sig == "bearish":           score -= 1

    if bb_pos == "below_lower":   score += 2
    elif bb_pos == "near_lower":  score += 1
    elif bb_pos == "above_upper": score -= 2
    elif bb_pos == "near_upper":  score -= 1

    if ma_sig == "golden_cross":  score += 2
    elif ma_sig == "above_both":  score += 1
    elif ma_sig == "death_cross": score -= 2
    elif ma_sig == "below_both":  score -= 1

    return TechnicalAnalysis(
        ticker=ticker,
        current_price=price,
        price_change_1d=change,
        price_change_1d_pct=change_pct,
        rsi14=rsi14,
        macd_histogram=hist_now,
        macd_signal=macd_sig,
        bb_upper=bb_upper,
        bb_middle=bb_mid,
        bb_lower=bb_low,
        bb_position=bb_pos,
        sma50=sma50,
        sma200=sma200,
        ma_signal=ma_sig,
        overall_signal=_overall(score),
        score=score,
        shares=shares,
        buy_price=buy_price,
        unrealized_pnl=(price - buy_price) * shares if buy_price > 0 else 0.0,
        unrealized_pnl_pct=((price - buy_price) / buy_price * 100) if buy_price > 0 else 0.0,
    )
