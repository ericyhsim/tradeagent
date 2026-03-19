"""
agents/fundamental.py
─────────────────────
FundamentalAgent — uses yfinance ticker info for:
  • PE ratio context
  • Analyst price target vs current price
  • 52-week positioning
  • Earnings / revenue growth
  • Analyst recommendation consensus

Intentionally lower baseline confidence (0.40) because fundamentals
move slowly and are less predictive on intraday/weekly horizons.
Designed to confirm or dampen signals, not to originate them.
"""
import logging
import yfinance as yf

from agents.base import AgentView

log = logging.getLogger(__name__)

# Sector PE thresholds — above this PE we start flagging extended valuations
EXTENDED_PE = 40.0
# Analyst recommendation: 1=strong buy, 3=hold, 5=strong sell
REC_BULLISH_THRESHOLD = 2.5
REC_BEARISH_THRESHOLD = 3.5


class FundamentalAgent:
    name = "fundamental"

    def analyze(self, symbol: str) -> AgentView:
        try:
            return self._run(symbol)
        except Exception as e:
            log.error(f"FundamentalAgent {symbol}: {e}")
            return AgentView(agent=self.name, symbol=symbol, direction="neutral",
                             confidence=0.40, reasons=[f"Data unavailable: {e}"])

    def _run(self, symbol: str) -> AgentView:
        info = yf.Ticker(symbol).info
        current = info.get("currentPrice") or info.get("regularMarketPrice") or 0

        votes     = []
        reasons   = []
        data      = {}

        # ── Analyst price target ─────────────────────────────────
        target_mean = info.get("targetMeanPrice")
        if target_mean and current:
            upside = (target_mean - current) / current * 100
            data["analyst_target"]     = round(target_mean, 2)
            data["analyst_upside_pct"] = round(upside, 1)
            if upside > 10:
                votes.append("call")
                reasons.append(f"Analyst target: ${target_mean:.0f} ({upside:+.1f}% upside)")
            elif upside < -10:
                votes.append("put")
                reasons.append(f"Analyst target: ${target_mean:.0f} ({upside:+.1f}% downside)")
        else:
            data["analyst_target"]     = None
            data["analyst_upside_pct"] = 0

        # ── 52-week position ─────────────────────────────────────
        hi52 = info.get("fiftyTwoWeekHigh")
        lo52 = info.get("fiftyTwoWeekLow")
        if hi52 and lo52 and hi52 > lo52 and current:
            pos52 = (current - lo52) / (hi52 - lo52)
            data["week52_position"] = round(pos52, 3)
            if pos52 < 0.25:
                votes.append("call")
                reasons.append(f"Near 52-wk low (position={pos52:.0%})")
            elif pos52 > 0.90:
                votes.append("put")
                reasons.append(f"Near 52-wk high (position={pos52:.0%})")
        else:
            data["week52_position"] = 0.5

        # ── PE ratio ─────────────────────────────────────────────
        pe = info.get("trailingPE") or info.get("forwardPE")
        data["pe"] = round(pe, 1) if pe else None
        if pe:
            if pe > EXTENDED_PE:
                votes.append("put")
                reasons.append(f"Extended PE: {pe:.0f}x")
            elif pe < 15:
                votes.append("call")
                reasons.append(f"Value PE: {pe:.0f}x")

        # ── Earnings growth ──────────────────────────────────────
        eg = info.get("earningsGrowth")
        rg = info.get("revenueGrowth")
        data["earnings_growth"] = round(eg, 3) if eg else None
        data["revenue_growth"]  = round(rg, 3) if rg else None
        if eg is not None:
            if eg > 0.20:
                votes.append("call")
                reasons.append(f"Strong earnings growth: {eg*100:.0f}%")
            elif eg < -0.10:
                votes.append("put")
                reasons.append(f"Declining earnings: {eg*100:.0f}%")

        # ── Analyst recommendation ────────────────────────────────
        rec = info.get("recommendationMean")
        data["analyst_recommendation"] = round(rec, 2) if rec else None
        if rec:
            if rec <= REC_BULLISH_THRESHOLD:
                votes.append("call")
                reasons.append(f"Analyst consensus: Buy ({rec:.1f}/5)")
            elif rec >= REC_BEARISH_THRESHOLD:
                votes.append("put")
                reasons.append(f"Analyst consensus: Sell ({rec:.1f}/5)")

        # ── Tally votes ──────────────────────────────────────────
        calls = votes.count("call")
        puts  = votes.count("put")
        total = calls + puts

        if total == 0:
            direction = "neutral"
            conf = 0.40
            reasons.append("No strong fundamental signal")
        else:
            direction = "call" if calls >= puts else "put"
            agreement = max(calls, puts) / total
            conf = 0.40 + agreement * 0.10 * min(total, 4)
            conf = min(conf, 0.75)

        if not reasons:
            reasons.append(f"Fundamental analysis: {calls}↑ {puts}↓ votes")

        return AgentView(
            agent=self.name, symbol=symbol,
            direction=direction, confidence=round(conf, 3),
            reasons=reasons, data=data,
        )
