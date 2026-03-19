"""
agents/indicators.py
────────────────────
Shared technical indicator helpers used by all agents.
Pure functions — no side effects, no imports from this project.
"""
import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta  = closes.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_g  = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l  = loss.ewm(com=period - 1, adjust=False).mean()
    rs     = avg_g / avg_l
    return 100 - 100 / (1 + rs)


def macd(closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram) as pd.Series."""
    fast_ema   = ema(closes, fast)
    slow_ema   = ema(closes, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def bb_bands(closes: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Returns (upper, middle, lower) Bollinger bands."""
    mid   = closes.rolling(period).mean()
    sigma = closes.rolling(period).std()
    return mid + std_dev * sigma, mid, mid - std_dev * sigma


def bb_pct_b(closes: pd.Series, period: int = 20) -> pd.Series:
    """Position within Bollinger bands (0=lower, 1=upper)."""
    upper, _, lower = bb_bands(closes, period)
    width = upper - lower
    return ((closes - lower) / width).clip(0, 1)


def bb_width_rank(closes: pd.Series, period: int = 20) -> pd.Series:
    """Percentile rank of BB width (low = squeeze)."""
    upper, mid, lower = bb_bands(closes, period)
    width = (upper - lower) / mid
    return width.rank(pct=True)


def z_score(series: pd.Series, window: int = 20) -> pd.Series:
    mu  = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mu) / std.replace(0, np.nan)


def rate_of_change(closes: pd.Series, periods: int = 5) -> pd.Series:
    return (closes - closes.shift(periods)) / closes.shift(periods)
