"""
agents/options_flow.py
──────────────────────
OptionsFlowAgent — options market sentiment.

Signals:
  • Put/call volume ratio
  • Put/call open interest ratio
  • IV skew (OTM put IV vs OTM call IV)
  • Unusual volume vs open interest (any strike vol > 3x OI)

Max natural confidence: 0.65
"""
import logging
from datetime import datetime, timedelta

import yfinance as yf

from agents.base import AgentView, _safe

log = logging.getLogger(__name__)


class OptionsFlowAgent:
    name = "options_flow"

    def analyze(self, symbol: str, quote: dict | None = None) -> AgentView:
        try:
            return self._run(symbol, quote)
        except Exception as e:
            log.error(f"OptionsFlowAgent {symbol}: {e}")
            return AgentView(
                agent=self.name,
                symbol=symbol,
                direction="neutral",
                confidence=0.0,
                reasons=[f"Analysis error: {e}"],
            )

    def _run(self, symbol: str, quote: dict | None) -> AgentView:
        ticker      = yf.Ticker(symbol)
        expirations = ticker.options

        if not expirations:
            return AgentView(
                agent=self.name,
                symbol=symbol,
                direction="neutral",
                confidence=0.40,
                reasons=["No options data"],
            )

        # ── Pick expiration 14-45 DTE ────────────────────────────────
        today      = datetime.now().date()
        target_exp = None
        for exp_str in expirations:
            try:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                dte      = (exp_date - today).days
                if 14 <= dte <= 45:
                    target_exp = exp_str
                    break
            except ValueError:
                continue

        # Fallback: skip nearest weekly (index 0), use index 1
        if target_exp is None:
            target_exp = expirations[1] if len(expirations) > 1 else expirations[0]

        chain = ticker.option_chain(target_exp)
        calls = chain.calls
        puts  = chain.puts

        if calls.empty or puts.empty:
            return AgentView(
                agent=self.name,
                symbol=symbol,
                direction="neutral",
                confidence=0.40,
                reasons=["Empty options chain"],
            )

        # ── Volume & OI ratios ───────────────────────────────────────
        call_vol   = calls["volume"].fillna(0).sum()
        put_vol    = puts["volume"].fillna(0).sum()
        pc_vol_ratio = put_vol / call_vol if call_vol > 0 else 1.0

        call_oi    = calls["openInterest"].fillna(0).sum()
        put_oi     = puts["openInterest"].fillna(0).sum()
        pc_oi_ratio  = put_oi / call_oi if call_oi > 0 else 1.0

        # ── Current price ────────────────────────────────────────────
        try:
            if quote and quote.get("c"):
                current_price = float(quote["c"])
            else:
                current_price = float(ticker.fast_info["last_price"])
        except Exception:
            # Fall back to midpoint of ATM strikes if price unavailable
            all_strikes   = calls["strike"].tolist()
            current_price = float(all_strikes[len(all_strikes) // 2]) if all_strikes else 100.0

        # ── ATM slice: within 5% of current price ───────────────────
        atm_calls = calls[
            calls["strike"].between(current_price * 0.95, current_price * 1.05)
        ]
        atm_puts  = puts[
            puts["strike"].between(current_price * 0.95, current_price * 1.05)
        ]

        # ── IV skew ──────────────────────────────────────────────────
        if not atm_puts.empty and not atm_calls.empty:
            iv_put_mean  = atm_puts["impliedVolatility"].mean()
            iv_call_mean = atm_calls["impliedVolatility"].mean()
            iv_skew      = float(iv_put_mean - iv_call_mean) if (iv_put_mean == iv_put_mean and iv_call_mean == iv_call_mean) else 0.0
        else:
            iv_skew = 0.0

        # ── Unusual volume ratio ─────────────────────────────────────
        call_vol_max    = calls["volume"].fillna(0).max()
        call_oi_mean    = max(float(calls["openInterest"].fillna(1).mean()), 1.0)
        unusual_call    = float(call_vol_max / call_oi_mean)

        put_vol_max     = puts["volume"].fillna(0).max()
        put_oi_mean     = max(float(puts["openInterest"].fillna(1).mean()), 1.0)
        unusual_put     = float(put_vol_max / put_oi_mean)

        # ── Direction votes ──────────────────────────────────────────
        votes: list[tuple[str, str]] = []  # (direction, reason)

        if pc_vol_ratio < 0.70:
            votes.append(("call", f"Put/call vol {pc_vol_ratio:.2f} — call-heavy flow"))
        elif pc_vol_ratio > 1.40:
            votes.append(("put", f"Put/call vol {pc_vol_ratio:.2f} — put-heavy flow"))

        if pc_oi_ratio < 0.80:
            votes.append(("call", f"Put/call OI {pc_oi_ratio:.2f} — bullish positioning"))
        elif pc_oi_ratio > 1.30:
            votes.append(("put", f"Put/call OI {pc_oi_ratio:.2f} — bearish positioning"))

        if iv_skew > 0.06:
            votes.append(("put", f"IV skew {iv_skew:.2f} — expensive puts, downside hedge"))
        elif iv_skew < -0.04:
            votes.append(("call", f"IV skew {iv_skew:.2f} — expensive calls, upside speculation"))

        if unusual_call > 3.0:
            votes.append(("call", f"Unusual call volume ({unusual_call:.1f}x OI)"))
        if unusual_put > 3.0:
            votes.append(("put", f"Unusual put volume ({unusual_put:.1f}x OI)"))

        # ── Compose result ───────────────────────────────────────────
        if not votes:
            return AgentView(
                agent=self.name,
                symbol=symbol,
                direction="neutral",
                confidence=0.40,
                reasons=["No unusual options flow"],
                data={
                    "pc_vol_ratio":       _safe(round(pc_vol_ratio, 3)),
                    "pc_oi_ratio":        _safe(round(pc_oi_ratio, 3)),
                    "iv_skew":            _safe(round(iv_skew, 3)),
                    "unusual_call_ratio": _safe(round(unusual_call, 2)),
                    "unusual_put_ratio":  _safe(round(unusual_put, 2)),
                    "expiration":         target_exp,
                },
            )

        n_calls = sum(1 for v, _ in votes if v == "call")
        n_puts  = sum(1 for v, _ in votes if v == "put")

        if n_calls > n_puts:
            direction = "call"
        elif n_puts > n_calls:
            direction = "put"
        else:
            direction = "neutral"

        if direction == "neutral":
            confidence = 0.40
        else:
            agreement  = max(n_calls, n_puts) / len(votes)
            confidence = min(0.45 + agreement * 0.20, 0.65)

        reasons = [reason for _, reason in votes]

        data = {
            "pc_vol_ratio":       _safe(round(pc_vol_ratio, 3)),
            "pc_oi_ratio":        _safe(round(pc_oi_ratio, 3)),
            "iv_skew":            _safe(round(iv_skew, 3)),
            "unusual_call_ratio": _safe(round(unusual_call, 2)),
            "unusual_put_ratio":  _safe(round(unusual_put, 2)),
            "expiration":         target_exp,
        }

        return AgentView(
            agent=self.name,
            symbol=symbol,
            direction=direction,
            confidence=round(confidence, 3),
            reasons=reasons,
            data=data,
        )
