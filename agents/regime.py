"""
agents/regime.py
────────────────
RegimeDetectionAgent — classifies the current market regime.

Labels: trending_up | trending_down | ranging | volatile

Inputs: daily candles (for ADX + realized vol) + ^VIX from yfinance.
Cached 5 minutes — regime is shared context, not per-symbol.

Used by ArbitrationAgent to dynamically adjust agent weight multipliers.
"""

from dataclasses import dataclass, field
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import time
import datetime

log = logging.getLogger(__name__)

CACHE_TTL = 300  # 5 minutes

REGIME_WEIGHT_MULTIPLIERS: dict[str, dict[str, float]] = {
    "trending_up": {
        "momentum": 1.1, "fundamental": 0.9, "sentiment": 1.0,
        "quant": 0.9, "macro": 1.0, "options_flow": 0.9, "stat_arb": 0.8,
    },
    "trending_down": {
        "momentum": 1.1, "fundamental": 0.9, "sentiment": 1.0,
        "quant": 0.9, "macro": 1.0, "options_flow": 0.9, "stat_arb": 0.8,
    },
    "ranging": {
        "momentum": 0.9, "fundamental": 1.0, "sentiment": 0.9,
        "quant": 1.1, "macro": 0.8, "options_flow": 1.0, "stat_arb": 1.1,
    },
    "volatile": {
        "momentum": 0.8, "fundamental": 0.8, "sentiment": 0.8,
        "quant": 0.9, "macro": 1.1, "options_flow": 1.2, "stat_arb": 0.7,
    },
}


@dataclass
class RegimeView:
    label: str                              # "trending_up" | "trending_down" | "ranging" | "volatile"
    vix: float
    adx: float
    realized_vol: float
    description: str
    weight_multipliers: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


class RegimeDetectionAgent:
    _cache: RegimeView | None = None
    _cache_ts: float = 0.0

    def classify(self, candles_1d: "pd.DataFrame | None", quote: "dict | None" = None) -> RegimeView:
        """Classify current market regime with 5-minute TTL cache."""
        if (
            RegimeDetectionAgent._cache is not None
            and time.time() - RegimeDetectionAgent._cache_ts < CACHE_TTL
        ):
            return RegimeDetectionAgent._cache

        try:
            result = self._run(candles_1d)
        except Exception as exc:
            log.warning("RegimeDetectionAgent._run failed: %s — defaulting to ranging", exc)
            result = RegimeView(
                label="ranging",
                vix=20.0,
                adx=20.0,
                realized_vol=0.20,
                description="Default regime (detection failed)",
                weight_multipliers=REGIME_WEIGHT_MULTIPLIERS["ranging"],
            )

        RegimeDetectionAgent._cache = result
        RegimeDetectionAgent._cache_ts = time.time()
        return result

    def _run(self, candles_1d) -> RegimeView:
        # 1. Fetch VIX
        try:
            vix_df = yf.Ticker("^VIX").history(period="5d", interval="1d")
            vix = float(vix_df["Close"].iloc[-1]) if len(vix_df) >= 1 else 20.0
        except Exception as exc:
            log.warning("VIX fetch failed: %s", exc)
            vix = 20.0

        # 2. Compute ADX
        adx_val, trend_up = self._adx(candles_1d)

        # 3. Compute realized volatility
        rv = 0.20
        if candles_1d is not None and len(candles_1d) >= 21:
            try:
                if "close" in candles_1d.columns:
                    closes = candles_1d["close"]
                else:
                    closes = candles_1d["Close"]
                log_rets = np.log(closes / closes.shift(1)).dropna()
                rv = float(log_rets.tail(20).std() * np.sqrt(252))
            except Exception as exc:
                log.warning("Realized vol computation failed: %s", exc)
                rv = 0.20

        # 4. Classification
        if vix > 28 or rv > 0.55:
            label = "volatile"
        elif adx_val > 25 and trend_up:
            label = "trending_up"
        elif adx_val > 25 and not trend_up:
            label = "trending_down"
        else:
            label = "ranging"

        description = (
            f"VIX={vix:.1f}, ADX={adx_val:.1f}, RealVol={rv:.2%}, trend_up={trend_up}"
        )

        return RegimeView(
            label=label,
            vix=vix,
            adx=adx_val,
            realized_vol=rv,
            description=description,
            weight_multipliers=REGIME_WEIGHT_MULTIPLIERS[label],
        )

    def _adx(self, df, period: int = 14) -> tuple[float, bool]:
        """Standard Wilder ADX. Returns (adx_float, trend_up_bool)."""
        try:
            if df is None or len(df) < period * 2 + 1:
                return (20.0, True)

            # Normalise column names to lowercase
            cols = {c.lower(): c for c in df.columns}
            high  = df[cols["high"]]
            low   = df[cols["low"]]
            close = df[cols["close"]]

            # True Range
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)

            # Directional movement
            up_move   = high.diff()
            down_move = -low.diff()

            plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            plus_dm_s  = pd.Series(plus_dm,  index=df.index).ewm(com=period - 1, adjust=False).mean()
            minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(com=period - 1, adjust=False).mean()
            atr_s      = tr.ewm(com=period - 1, adjust=False).mean()

            plus_di  = 100 * plus_dm_s  / atr_s.replace(0, np.nan)
            minus_di = 100 * minus_dm_s / atr_s.replace(0, np.nan)

            dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
            adx = dx.ewm(com=period - 1, adjust=False).mean()

            adx_val = float(adx.iloc[-1])

            # Trend direction: last close vs 20-period EMA of close
            ema_20   = close.ewm(span=20, adjust=False).mean()
            trend_up = bool(close.iloc[-1] > ema_20.iloc[-1])

            return (adx_val, trend_up)

        except Exception as exc:
            log.warning("ADX computation failed: %s", exc)
            return (20.0, True)
