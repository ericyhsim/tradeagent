"""
agents/macro.py
───────────────
MacroAgent — macro environment analysis.

Signals:
  • VIX level + trend (fear/complacency)
  • Yield curve slope (10Y - 2Y via ^TNX and ^IRX)
  • Dollar strength trend (DX-Y.NYB 5-day change)
  • Risk-on/off sector rotation (XLY vs XLP performance)

Cached 5 minutes — same macro data reused across all symbols in a scan.
Max natural confidence: 0.65
"""
import logging
import time

import yfinance as yf

from agents.base import AgentView, _safe

log = logging.getLogger(__name__)


class MacroAgent:
    name = "macro"

    _macro_cache: dict = {}
    _cache_ts: float = 0.0

    CACHE_TTL = 300  # seconds

    def analyze(self, symbol: str) -> AgentView:
        try:
            macro = self._get_macro()

            vix             = macro["vix"]
            vix_5d_change   = macro["vix_5d_change"]
            yield_10y       = macro["yield_10y"]
            yield_2y        = macro["yield_2y"]
            dxy_chg_pct     = macro["dxy_chg_pct"]
            risk_on         = macro["risk_on"]

            votes: list[tuple[str, str]] = []  # (direction, reason)

            # ── VIX ──────────────────────────────────────────────────
            if vix > 28:
                votes.append(("put", f"VIX={vix:.1f} — elevated fear"))
            elif vix < 15 and vix_5d_change < 0:
                votes.append(("call", f"VIX={vix:.1f} falling — complacency"))
            elif vix_5d_change > 3:
                votes.append(("put", f"VIX rising +{vix_5d_change:.1f}"))

            # ── Yield curve ──────────────────────────────────────────
            curve = yield_10y - yield_2y
            if curve < 0:
                votes.append(("put", f"Inverted yield curve {curve*100:.0f}bps"))
            elif curve > 0.5:
                votes.append(("call", f"Normal curve +{curve*100:.0f}bps"))

            # ── Dollar ───────────────────────────────────────────────
            if dxy_chg_pct > 1.2:
                votes.append(("put", f"DXY +{dxy_chg_pct:.1f}% — dollar headwind"))
            elif dxy_chg_pct < -1.2:
                votes.append(("call", f"DXY {dxy_chg_pct:.1f}% — dollar tailwind"))

            # ── Sector rotation ──────────────────────────────────────
            if risk_on:
                votes.append(("call", "Risk-on rotation (XLY > XLP)"))
            else:
                votes.append(("put", "Risk-off rotation (XLP > XLY)"))

            # ── Direction by majority vote ────────────────────────────
            if not votes:
                direction = "neutral"
                confidence = 0.40
                reasons = ["No macro signal"]
            else:
                calls = sum(1 for v, _ in votes if v == "call")
                puts  = sum(1 for v, _ in votes if v == "put")

                if calls > puts:
                    direction = "call"
                elif puts > calls:
                    direction = "put"
                else:
                    direction = "neutral"

                if direction == "neutral":
                    confidence = 0.40
                else:
                    agreement  = max(calls, puts) / len(votes)
                    confidence = min(0.40 + agreement * 0.25, 0.65)

                reasons = [reason for _, reason in votes]

            data = {
                "vix":              _safe(round(vix, 2)),
                "vix_5d_change":    _safe(round(vix_5d_change, 2)),
                "yield_10y":        _safe(round(yield_10y, 3)),
                "yield_curve_bps":  _safe(round((yield_10y - yield_2y) * 100, 1)),
                "dxy_chg_pct":      _safe(round(dxy_chg_pct, 2)),
                "risk_on":          risk_on,
            }

            if "fetch_error" in macro:
                reasons.append(f"Macro fetch error (using defaults): {macro['fetch_error']}")

            return AgentView(
                agent=self.name,
                symbol=symbol,
                direction=direction,
                confidence=round(confidence, 3),
                reasons=reasons,
                data=data,
            )

        except Exception as e:
            log.error(f"MacroAgent {symbol}: {e}")
            return AgentView(
                agent=self.name,
                symbol=symbol,
                direction="neutral",
                confidence=0.0,
                reasons=[f"Analysis error: {e}"],
            )

    def _get_macro(self) -> dict:
        if MacroAgent._macro_cache and (time.time() - MacroAgent._cache_ts) < self.CACHE_TTL:
            return MacroAgent._macro_cache

        try:
            # VIX
            vix_df = yf.Ticker("^VIX").history(period="5d", interval="1d")
            vix         = float(vix_df["Close"].iloc[-1])
            vix_5d_change = float(vix_df["Close"].iloc[-1] - vix_df["Close"].iloc[0])

            # 10-year yield (^TNX quotes in tenths of a percent, e.g. 45 = 4.5%)
            tnx_df   = yf.Ticker("^TNX").history(period="10d", interval="1d")
            yield_10y = float(tnx_df["Close"].iloc[-1]) / 10.0

            # 2-year yield (^IRX quotes in hundredths of a percent, e.g. 450 = 4.5%)
            irx_df   = yf.Ticker("^IRX").history(period="5d", interval="1d")
            yield_2y  = float(irx_df["Close"].iloc[-1]) / 100.0

            # Dollar index — 5-day % change
            dxy_df    = yf.Ticker("DX-Y.NYB").history(period="10d", interval="1d")
            dxy_now   = float(dxy_df["Close"].iloc[-1])
            dxy_5ago  = float(dxy_df["Close"].iloc[-6]) if len(dxy_df) >= 6 else float(dxy_df["Close"].iloc[0])
            dxy_chg_pct = (dxy_now - dxy_5ago) / dxy_5ago * 100.0

            # Risk-on / risk-off: XLY vs XLP 5-day return
            xly_df   = yf.Ticker("XLY").history(period="5d", interval="1d")
            xlp_df   = yf.Ticker("XLP").history(period="5d", interval="1d")
            xly_ret  = (float(xly_df["Close"].iloc[-1]) - float(xly_df["Close"].iloc[0])) / float(xly_df["Close"].iloc[0])
            xlp_ret  = (float(xlp_df["Close"].iloc[-1]) - float(xlp_df["Close"].iloc[0])) / float(xlp_df["Close"].iloc[0])
            risk_on  = bool(xly_ret > xlp_ret)

            result = {
                "vix":            vix,
                "vix_5d_change":  vix_5d_change,
                "yield_10y":      yield_10y,
                "yield_2y":       yield_2y,
                "dxy_chg_pct":    dxy_chg_pct,
                "risk_on":        risk_on,
            }

        except Exception as e:
            log.warning(f"MacroAgent fetch error, using defaults: {e}")
            result = {
                "vix":            20.0,
                "vix_5d_change":  0.0,
                "yield_10y":      4.5,
                "yield_2y":       4.5,
                "dxy_chg_pct":    0.0,
                "risk_on":        True,
                "fetch_error":    str(e),
            }

        MacroAgent._macro_cache = result
        MacroAgent._cache_ts    = time.time()
        return result
