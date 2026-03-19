"""
agents/stat_arb.py
──────────────────
StatArbAgent — statistical arbitrage via pair spread Z-score.

If a correlation pair is defined for the symbol, computes the log-ratio
spread Z-score vs the correlated peer using 60-day daily candles.
Falls back to standalone 20-period price Z-score if no pair defined.

Max natural confidence: 0.70
"""
import logging

import numpy as np
import pandas as pd
import yfinance as yf

from agents.base       import AgentView, _safe
from agents.indicators import z_score

log = logging.getLogger(__name__)

PAIRS: dict[str, str] = {
    "AAPL": "MSFT", "MSFT": "AAPL",
    "NVDA": "AMD",  "AMD": "NVDA",
    "GOOGL": "META", "META": "GOOGL",
    "SPY": "QQQ",   "QQQ": "SPY",
    "COIN": "MSTR", "MSTR": "COIN",
    "TSLA": "RIVN", "RIVN": "TSLA",
    "AMZN": "SHOP", "SHOP": "AMZN",
    "ORCL": "CRM",  "CRM": "ORCL",
    "SOFI": "HOOD", "HOOD": "SOFI",
    "PLTR": "ARKK", "ARM": "NVDA",
    "INTC": "AMD",  "ADBE": "CRM",
}


def _get_close(df: pd.DataFrame) -> pd.Series:
    """Return the close column regardless of capitalisation."""
    if "close" in df.columns:
        return df["close"]
    return df["Close"]


class StatArbAgent:
    name = "stat_arb"

    def analyze(
        self,
        symbol: str,
        candles_1d: "pd.DataFrame | None" = None,
    ) -> AgentView:
        try:
            return self._run(symbol, candles_1d)
        except Exception as e:
            log.error(f"StatArbAgent {symbol}: {e}")
            return AgentView(
                agent=self.name,
                symbol=symbol,
                direction="neutral",
                confidence=0.0,
                reasons=[f"Analysis error: {e}"],
            )

    def _run(self, symbol: str, candles_1d: "pd.DataFrame | None") -> AgentView:
        sym_upper = symbol.upper()
        pair      = PAIRS.get(sym_upper)

        z_val  = 0.0
        corr   = 1.0
        method = "standalone_zscore"

        # ── Pair spread path ─────────────────────────────────────────
        if pair:
            try:
                # Symbol candles
                if candles_1d is not None and len(candles_1d) >= 20:
                    sym_df = candles_1d
                else:
                    sym_df = yf.Ticker(symbol).history(period="60d", interval="1d")

                pair_df = yf.Ticker(pair).history(period="60d", interval="1d")

                sym_close  = _get_close(sym_df)
                pair_close = _get_close(pair_df)

                # Align to the same length
                n = min(len(sym_close), len(pair_close))

                if n >= 15:
                    sym_close  = sym_close.iloc[-n:].reset_index(drop=True)
                    pair_close = pair_close.iloc[-n:].reset_index(drop=True)

                    log_ratio = np.log(sym_close.values / pair_close.values)
                    spread    = pd.Series(log_ratio)

                    window  = min(20, n - 1)
                    z_series = z_score(spread, window)
                    z_raw    = z_series.iloc[-1]
                    z_val    = 0.0 if (z_raw != z_raw) else float(z_raw)  # NaN → 0.0

                    corr_matrix = np.corrcoef(sym_close.values, pair_close.values)
                    corr        = float(corr_matrix[0, 1])
                    if corr != corr:  # NaN guard
                        corr = 0.5

                    method = "log_ratio_pair"
                else:
                    # Not enough overlapping data — fall through to standalone
                    pair = None

            except Exception as e:
                log.warning(f"StatArbAgent {symbol} pair fetch failed ({pair}): {e}")
                pair = None

        # ── Standalone Z-score path ──────────────────────────────────
        if method == "standalone_zscore" or pair is None:
            pair   = None
            method = "standalone_zscore"
            corr   = 1.0

            try:
                if candles_1d is not None and len(candles_1d) >= 20:
                    closes = _get_close(candles_1d)
                else:
                    hist   = yf.Ticker(symbol).history(period="60d", interval="1d")
                    closes = _get_close(hist)

                z_series = z_score(closes, 20)
                z_raw    = z_series.iloc[-1]
                z_val    = 0.0 if (z_raw != z_raw) else float(z_raw)

            except Exception as e:
                log.warning(f"StatArbAgent {symbol} standalone fetch failed: {e}")
                z_val = 0.0

        # ── Direction from Z-score — graduated thresholds ────────────
        abs_z = abs(z_val)
        pair_lbl = pair or symbol

        if z_val < -2.0:
            direction = "call"
            reason = (f"{symbol} cheap vs {pair_lbl} (Z={z_val:.2f}) — strong mean-reversion long"
                      if method == "log_ratio_pair" else
                      f"{symbol} Z={z_val:.2f} — statistically oversold")
        elif z_val > 2.0:
            direction = "put"
            reason = (f"{symbol} expensive vs {pair_lbl} (Z={z_val:.2f}) — strong mean-reversion short"
                      if method == "log_ratio_pair" else
                      f"{symbol} Z={z_val:.2f} — statistically overbought")
        elif z_val < -1.3:
            direction = "call"
            reason = (f"{symbol} vs {pair_lbl} spread Z={z_val:.2f} — moderate reversion lean long"
                      if method == "log_ratio_pair" else
                      f"{symbol} Z={z_val:.2f} — mild oversold lean")
        elif z_val > 1.3:
            direction = "put"
            reason = (f"{symbol} vs {pair_lbl} spread Z={z_val:.2f} — moderate reversion lean short"
                      if method == "log_ratio_pair" else
                      f"{symbol} Z={z_val:.2f} — mild overbought lean")
        else:
            direction = "neutral"
            reason = (f"{symbol}/{pair_lbl} spread Z={z_val:.2f} — within normal range"
                      if method == "log_ratio_pair" else
                      f"{symbol} Z={z_val:.2f} — no statistical edge")

        # ── Confidence ────────────────────────────────────────────────
        corr_factor = max(corr, 0.5) if corr > 0 else 0.5
        if abs_z >= 2.0:
            z_strength  = min(abs_z - 2.0, 2.0) / 2.0
            confidence  = min(0.52 + z_strength * 0.18 * corr_factor, 0.70)
        elif abs_z >= 1.3:
            z_strength  = (abs_z - 1.3) / 0.7
            confidence  = 0.43 + z_strength * 0.09 * corr_factor
        else:
            confidence  = 0.40

        data = {
            "pair":        pair,
            "spread_z":    _safe(round(z_val, 3)),
            "correlation": _safe(round(corr, 3)),
            "method":      method,
        }

        return AgentView(
            agent=self.name,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            reasons=[reason],
            data=data,
        )
