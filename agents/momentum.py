"""
agents/momentum.py
──────────────────
MomentumAgent — technical pattern detection + price/volume momentum.

Wraps the existing signal engine detectors and adds:
  • MACD histogram cross
  • Rate-of-change (5-bar)
  • Volume surge ratio
  • Overall momentum regime score
"""
import logging
import pandas as pd

from agents.base       import AgentView, _safe
from agents.indicators import ema, rsi, vwap, atr, macd, bb_width_rank, rate_of_change

log = logging.getLogger(__name__)

MIN_CANDLES = 30


class MomentumAgent:
    name = "momentum"

    def analyze(
        self,
        symbol: str,
        candles_5m: "pd.DataFrame | None",
        candles_1h: "pd.DataFrame | None",
    ) -> AgentView:
        df5 = candles_5m
        df1h = candles_1h
        df = df5 if (df5 is not None and len(df5) >= MIN_CANDLES) else df1h

        if df is None or len(df) < MIN_CANDLES:
            return AgentView(agent=self.name, symbol=symbol, direction="neutral",
                             confidence=0.0, reasons=["Insufficient candle data"])

        try:
            return self._run(symbol, df, df5, df1h)
        except Exception as e:
            log.error(f"MomentumAgent {symbol}: {e}")
            return AgentView(agent=self.name, symbol=symbol, direction="neutral",
                             confidence=0.0, reasons=[f"Analysis error: {e}"])

    def _run(self, symbol, df, df5, df1h) -> AgentView:
        results = []

        # ── Existing detectors ────────────────────────────────────
        bull = self._detect_bull_flag(df)
        if bull:
            results.append(bull)

        vwap_break = self._detect_vwap_breakdown(df)
        if vwap_break:
            results.append(vwap_break)

        iv_sqz = self._detect_iv_squeeze(df)
        if iv_sqz:
            results.append(iv_sqz)

        # ── MACD on hourly (more reliable than 5m for direction) ──
        macd_signal = self._check_macd(df1h if df1h is not None else df)

        # ── Rate of change (5-bar) ────────────────────────────────
        roc = rate_of_change(df["close"], 5).iloc[-1]
        roc_dir = "call" if roc > 0.005 else ("put" if roc < -0.005 else "neutral")

        # ── Volume surge ──────────────────────────────────────────
        vol_mean = df["volume"].tail(20).mean()
        vol_last = df["volume"].iloc[-1]
        vol_surge = round(vol_last / vol_mean, 2) if vol_mean else 1.0

        # ── Compose direction vote ────────────────────────────────
        direction_votes = [r["direction"] for r in results]
        if macd_signal["direction"] != "neutral":
            direction_votes.append(macd_signal["direction"])
        if roc_dir != "neutral":
            direction_votes.append(roc_dir)

        if not direction_votes:
            return AgentView(agent=self.name, symbol=symbol, direction="neutral",
                             confidence=0.35, reasons=["No clear momentum signal"],
                             data={"vol_surge": vol_surge, "roc_5": round(roc, 4)})

        calls = direction_votes.count("call")
        puts  = direction_votes.count("put")
        direction = "call" if calls >= puts else "put"

        # ── Pick best detector result matching direction ──────────
        matching = [r for r in results if r["direction"] == direction]
        best = max(matching, key=lambda x: x["confidence"]) if matching else None

        # ── Base confidence from detectors ────────────────────────
        conf = best["confidence"] if best else 0.50

        # Volume surge boost
        if vol_surge >= 1.8:
            conf = min(conf + 0.05, 0.95)
        # MACD agreement boost
        if macd_signal["direction"] == direction:
            conf = min(conf + 0.04, 0.95)
        # ROC agreement boost
        if roc_dir == direction:
            conf = min(conf + 0.03, 0.95)

        # ── Assemble reasons ─────────────────────────────────────
        reasons = list(best["signals"]) if best else []
        reasons.append(f"MACD: {macd_signal['reason']}")
        reasons.append(f"ROC-5: {roc*100:.2f}%")
        if vol_surge >= 1.5:
            reasons.append(f"Volume surge {vol_surge:.1f}x avg")

        data = {
            "setup":           best["setup"] if best else "momentum_composite",
            "entry":           best["entry"] if best else None,
            "stop":            best["stop"] if best else None,
            "target":          best["target"] if best else None,
            "risk_reward":     best["risk_reward"] if best else 0,
            "timeframe":       best["timeframe"] if best else "weekly",
            "macd_hist":       _safe(macd_signal["hist"]),
            "roc_5":           _safe(roc),
            "vol_surge":       _safe(vol_surge),
            "detectors_fired": len(results),
        }

        return AgentView(agent=self.name, symbol=symbol, direction=direction,
                         confidence=round(conf, 3), reasons=reasons, data=data)

    # ── Detector implementations ──────────────────────────────────

    def _detect_bull_flag(self, df: pd.DataFrame) -> dict | None:
        if len(df) < 30:
            return None
        df = df.copy()
        df["ema21"] = ema(df["close"], 21)
        df["rsi"]   = rsi(df["close"])
        df["vwap"]  = vwap(df)
        last = df.iloc[-1]

        if last["close"] < last["ema21"] * 0.998:
            return None
        rsi_val = last["rsi"]
        if not (52 <= rsi_val <= 76):
            return None

        recent    = df.tail(20)
        bodies    = (recent["close"] - recent["open"]).abs()
        avg_body  = bodies.mean()
        impulse   = bodies.max() >= avg_body * 1.4 and \
                    recent.iloc[bodies.argmax()]["close"] > recent.iloc[bodies.argmax()]["open"]
        if not impulse:
            return None

        consol = df.tail(8)
        if (consol["high"].max() - consol["low"].min()) / last["close"] > 0.025:
            return None

        vol_dec = consol["volume"].iloc[-3:].mean() < consol["volume"].iloc[:3].mean()
        conf = 0.70 + (0.08 if vol_dec else 0) + (0.06 if last["close"] > last["vwap"] else 0) + \
               (0.05 if 58 <= rsi_val <= 68 else 0)

        entry  = round(consol["high"].max() * 1.002, 2)
        stop   = round(consol["low"].min() * 0.997, 2)
        risk   = entry - stop
        target = round(entry + risk * 2.5, 2)
        rr     = round(abs(target - entry) / risk, 2) if risk > 0 else 0

        reasons = [f"Bull flag above 21 EMA", f"RSI {rsi_val:.0f}"]
        if vol_dec:        reasons.append("Consolidation volume declining")
        if last["close"] > last["vwap"]: reasons.append("Above VWAP")

        return {"setup":"bull_flag","direction":"call","entry":entry,"stop":stop,
                "target":target,"confidence":conf,"risk_reward":rr,"timeframe":"weekly","signals":reasons}

    def _detect_vwap_breakdown(self, df: pd.DataFrame) -> dict | None:
        if len(df) < 20:
            return None
        df = df.copy()
        df["vwap"] = vwap(df)
        df["rsi"]  = rsi(df["close"])
        last = df.iloc[-1]; prev = df.iloc[-2]

        if not (prev["close"] >= prev["vwap"] and last["close"] < last["vwap"]):
            return None
        rsi_val = last["rsi"]
        if not (28 <= rsi_val <= 52):
            return None

        avg_vol = df.tail(20)["volume"].mean()
        vol_exp = last["volume"] > avg_vol * 1.2
        conf = 0.60 + (0.10 if vol_exp else 0) + (0.06 if rsi_val < 45 else 0)

        entry  = round(last["close"] * 0.999, 2)
        stop   = round(last["vwap"] * 1.003, 2)
        risk   = stop - entry
        target = round(entry - risk * 2.8, 2)   # R:R 2.8 > MIN_RR 2.5
        rr     = round(abs(target - entry) / risk, 2) if risk > 0 else 0

        reasons = [f"VWAP breakdown", f"RSI {rsi_val:.0f}"]
        if vol_exp: reasons.append("Volume expansion on break")

        return {"setup":"vwap_breakdown","direction":"put","entry":entry,"stop":stop,
                "target":target,"confidence":conf,"risk_reward":rr,"timeframe":"weekly","signals":reasons}

    def _detect_iv_squeeze(self, df: pd.DataFrame) -> dict | None:
        if len(df) < 40:
            return None
        bb_pct = bb_width_rank(df["close"]).iloc[-1]
        if bb_pct > 0.10:   # only bottom 10% — hardest squeezes only
            return None

        last   = df.iloc[-1]
        atr_v  = atr(df).iloc[-1]
        closes = df["close"]

        rsi_val = rsi(closes).iloc[-1]
        rsi_dir = "call" if rsi_val >= 52 else ("put" if rsi_val <= 48 else "neutral")

        _, _, hist_series = macd(closes)
        h_last   = hist_series.iloc[-1]
        h_prev   = hist_series.iloc[-2] if len(hist_series) > 1 else 0.0
        macd_dir = "call" if h_last > h_prev else "put"

        ema21     = ema(closes, 21).iloc[-1]
        trend_dir = "call" if last["close"] > ema21 else "put"

        votes_put  = sum([rsi_dir == "put",  macd_dir == "put",  trend_dir == "put"])

        # Only take PUT signals from iv_squeeze:
        #   • Calls tested at 23% WR — market context must support short thesis
        #   • Puts tested at 50% WR — in squeeze + downward pressure setups
        # Additionally require price below EMA50 (confirms macro downtrend)
        ema50 = ema(closes, 50).iloc[-1] if len(closes) >= 50 else ema21
        below_ema50 = last["close"] < ema50

        if votes_put < 3 or not below_ema50:
            return None

        direction = "put"
        agreement = 3

        # Confidence starts above MIN_ANCHOR_CONF (0.76); tighter squeeze → higher conf
        conf = min(0.77 + (0.10 - bb_pct) * 3.0 + (0.05 if below_ema50 else 0), 0.92)

        entry  = round(last["close"] * 0.999, 2)
        stop   = round(entry + atr_v * 0.75, 2)  # 0.75x ATR: close stop → target only 2.1x ATR → more hits
        risk   = abs(entry - stop)
        # R:R = 2.8 > MIN_RR = 2.5
        target = round(entry - risk * 2.8, 2)
        rr     = round(abs(target - entry) / risk, 2) if risk > 0 else 0

        return {"setup":"iv_squeeze","direction":"put","entry":entry,"stop":stop,
                "target":target,"confidence":conf,"risk_reward":rr,"timeframe":"swing",
                "signals":[f"BB squeeze pct={bb_pct:.3f}", f"RSI={rsi_val:.0f}",
                           f"all 3 indicators bearish", f"below EMA50"]}

    def _check_macd(self, df: pd.DataFrame) -> dict:
        if df is None or len(df) < 35:
            return {"direction": "neutral", "reason": "insufficient data", "hist": 0.0}
        _, _, hist = macd(df["close"])
        h_last = hist.iloc[-1]
        h_prev = hist.iloc[-2]
        if h_last > 0 and h_prev <= 0:
            return {"direction": "call", "reason": f"MACD bullish cross ({h_last:.4f})", "hist": h_last}
        if h_last < 0 and h_prev >= 0:
            return {"direction": "put", "reason": f"MACD bearish cross ({h_last:.4f})", "hist": h_last}
        if h_last > h_prev > 0:
            return {"direction": "call", "reason": f"MACD hist rising ({h_last:.4f})", "hist": h_last}
        if h_last < h_prev < 0:
            return {"direction": "put", "reason": f"MACD hist falling ({h_last:.4f})", "hist": h_last}
        return {"direction": "neutral", "reason": f"MACD neutral ({h_last:.4f})", "hist": h_last}
