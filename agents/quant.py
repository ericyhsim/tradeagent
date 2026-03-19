"""
agents/quant.py
───────────────
QuantAgent — statistical / quantitative analysis.

Signals:
  • Returns Z-score (mean reversion vs breakout)
  • Bollinger %B positioning
  • Volume Z-score
  • Realized volatility (20-day)
  • ATR% (risk context)

Designed as a confirmation agent: max natural confidence ~0.68.
Identifies statistical extremes that support or oppose a trade.
"""
import logging
import numpy as np
import pandas as pd

from agents.base       import AgentView, _safe
from agents.indicators import z_score, bb_pct_b, atr

log = logging.getLogger(__name__)

MIN_CANDLES_DAILY = 25
MIN_CANDLES_INTRA = 20


class QuantAgent:
    name = "quant"

    def analyze(
        self,
        symbol: str,
        candles_1d: "pd.DataFrame | None",
        candles_5m: "pd.DataFrame | None",
    ) -> AgentView:
        try:
            return self._run(symbol, candles_1d, candles_5m)
        except Exception as e:
            log.error(f"QuantAgent {symbol}: {e}")
            return AgentView(agent=self.name, symbol=symbol, direction="neutral",
                             confidence=0.40, reasons=[f"Analysis error: {e}"])

    def _run(self, symbol, df_d, df_5m) -> AgentView:
        df = df_d if (df_d is not None and len(df_d) >= MIN_CANDLES_DAILY) else df_5m
        if df is None or len(df) < MIN_CANDLES_INTRA:
            return AgentView(agent=self.name, symbol=symbol, direction="neutral",
                             confidence=0.40, reasons=["Insufficient data"])

        closes  = df["close"]
        volumes = df["volume"]

        # ── Returns Z-score ──────────────────────────────────────
        log_rets = np.log(closes / closes.shift(1)).dropna()
        price_z  = float(z_score(closes, 20).iloc[-1]) if len(closes) >= 20 else 0.0
        if np.isnan(price_z): price_z = 0.0

        # ── Bollinger %B ─────────────────────────────────────────
        bb_b = float(bb_pct_b(closes, 20).iloc[-1]) if len(closes) >= 20 else 0.5
        if np.isnan(bb_b): bb_b = 0.5

        # ── Volume Z-score ───────────────────────────────────────
        vol_z = float(z_score(volumes, 20).iloc[-1]) if len(volumes) >= 20 else 0.0
        if np.isnan(vol_z): vol_z = 0.0

        # ── Realized volatility (annualized) ─────────────────────
        rv = float(log_rets.tail(20).std() * np.sqrt(252)) if len(log_rets) >= 20 else 0.0

        # ── ATR% ─────────────────────────────────────────────────
        atr_val  = float(atr(df).iloc[-1]) if len(df) >= 15 else 0.0
        last_px  = float(closes.iloc[-1])
        atr_pct  = round(atr_val / last_px, 4) if last_px else 0.0

        # ── Direction: graduated signal strength ─────────────────
        # Strong: both Z and BB%B confirm the extreme
        # Soft:   either Z or BB%B alone shows a lean
        strong_oversold   = price_z < -1.5 and bb_b < 0.20
        strong_overbought = price_z >  1.5 and bb_b > 0.80
        soft_oversold     = price_z < -1.0 or  bb_b < 0.12
        soft_overbought   = price_z >  1.0 or  bb_b > 0.88

        reasons = []

        if strong_oversold:
            direction = "call"
            reasons.append(f"Strong oversold extreme — Z={price_z:.2f}, BB%B={bb_b:.2f}")
        elif strong_overbought:
            direction = "put"
            reasons.append(f"Strong overbought extreme — Z={price_z:.2f}, BB%B={bb_b:.2f}")
        elif soft_oversold and not soft_overbought:
            direction = "call"
            reasons.append(f"Soft oversold lean — Z={price_z:.2f}, BB%B={bb_b:.2f}")
        elif soft_overbought and not soft_oversold:
            direction = "put"
            reasons.append(f"Soft overbought lean — Z={price_z:.2f}, BB%B={bb_b:.2f}")
        else:
            direction = "neutral"
            reasons.append(f"Mid-range — Z={price_z:.2f}, BB%B={bb_b:.2f}")

        if vol_z > 2.0:
            reasons.append(f"Volume surge: Z={vol_z:.1f} — breakout/event risk")
        elif vol_z < -1.5:
            reasons.append(f"Low volume: Z={vol_z:.1f} — reduced conviction")

        if rv > 0.60:
            reasons.append(f"High realized vol {rv*100:.0f}% — wider stops advised")
        elif rv < 0.20:
            reasons.append(f"Low realized vol {rv*100:.0f}% — tight market")

        if atr_pct > 0.03:
            reasons.append(f"ATR% {atr_pct*100:.1f}% — elevated intraday range")

        # ── Confidence ───────────────────────────────────────────
        if strong_oversold or strong_overbought:
            z_strength = min(abs(price_z), 3.0) / 3.0
            conf = 0.52 + z_strength * 0.16
        elif direction != "neutral":
            z_strength = min(abs(price_z), 2.0) / 2.0
            conf = 0.44 + z_strength * 0.12
        else:
            conf = 0.40
        conf = round(min(conf, 0.68), 3)

        return AgentView(
            agent=self.name, symbol=symbol,
            direction=direction, confidence=conf,
            reasons=reasons,
            data={
                "price_z_score":  _safe(round(price_z, 3)),
                "bb_pct_b":       _safe(round(bb_b, 3)),
                "volume_z_score": _safe(round(vol_z, 3)),
                "realized_vol":   _safe(round(rv, 3)),
                "atr_pct":        _safe(atr_pct),
            },
        )
