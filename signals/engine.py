"""
signals/engine.py
─────────────────
Processes Finnhub data through pattern detectors and produces
high-probability options trade alerts.
"""

import logging
import pandas as pd
from dataclasses import dataclass, field, asdict
from datetime import datetime

log = logging.getLogger(__name__)

MIN_CONFIDENCE = 0.70
MIN_RR = 2.0


@dataclass
class TradeAlert:
    symbol: str
    setup: str
    direction: str          # 'call' | 'put'
    entry: float
    stop: float
    target: float
    confidence: float
    risk_reward: float
    timeframe: str          # 'weekly' | 'swing'
    signals: list           = field(default_factory=list)
    timestamp: str          = field(default_factory=lambda: datetime.now().isoformat())
    # Multi-agent enrichment fields (populated by Orchestrator)
    signal_id:       str    = field(default="")
    agent_views:     list   = field(default_factory=list)
    consensus_score: float  = 0.0
    agent_weights:   dict   = field(default_factory=dict)

    def to_dict(self):
        from agents.base import _clean
        d = asdict(self)
        d["risk_reward"]    = round(d["risk_reward"], 2)
        d["confidence_pct"] = round(d["confidence"] * 100)
        d["consensus_pct"]  = round(d["consensus_score"] * 100)
        return _clean(d)


# ── Indicator helpers (aliases to shared module) ──────────────────

from agents.indicators import (
    ema   as _ema_fn,
    rsi   as _rsi_fn,
    vwap  as _vwap_fn,
    atr   as _atr_fn,
    bb_width_rank as _bb_width_pct_fn,
)

def _ema(series, span):       return _ema_fn(series, span)
def _rsi(closes, period=14):  return _rsi_fn(closes, period)
def _vwap(df):                return _vwap_fn(df)
def _atr(df, period=14):      return _atr_fn(df, period)
def _bb_width_pct(closes, period=20): return _bb_width_pct_fn(closes, period)

def is_chop_day(df, vix=None):
    if df is None or len(df) < 10:
        return False
    today = df.tail(78)
    day_range = (today["high"].max() - today["low"].min()) / today["close"].mean()
    atr_v = _atr(today).iloc[-1] / today["close"].mean()
    if vix and vix < 13:
        return True
    return day_range < 0.008 and atr_v < 0.003


# ── Detectors ─────────────────────────────────

def detect_bull_flag(df):
    if df is None or len(df) < 30:
        return None
    df = df.copy()
    df["ema21"] = _ema(df["close"], 21)
    df["rsi"] = _rsi(df["close"])
    df["vwap"] = _vwap(df)
    last = df.iloc[-1]

    if last["close"] < last["ema21"] * 0.998:
        return None
    rsi_val = last["rsi"]
    if not (52 <= rsi_val <= 76):
        return None

    recent = df.tail(20)
    body_sizes = (recent["close"] - recent["open"]).abs()
    avg_body = body_sizes.mean()
    max_idx = body_sizes.argmax()
    max_candle = recent.iloc[max_idx]
    had_impulse = (max_candle["close"] > max_candle["open"] and
                   body_sizes.max() >= avg_body * 1.4)
    if not had_impulse:
        return None

    consol = df.tail(8)
    consol_range = (consol["high"].max() - consol["low"].min()) / last["close"]
    if consol_range > 0.025:
        return None

    vol_declining = consol["volume"].iloc[-3:].mean() < consol["volume"].iloc[:3].mean()

    confidence = 0.62
    confidence += 0.08 if vol_declining else 0
    confidence += 0.06 if last["close"] > last["vwap"] else 0
    confidence += 0.05 if 58 <= rsi_val <= 68 else 0

    entry = round(consol["high"].max() * 1.002, 2)
    stop = round(consol["low"].min() * 0.997, 2)
    risk = entry - stop
    target = round(entry + risk * 2.5, 2)

    reasons = [f"Bull flag above 21 EMA", f"RSI {rsi_val:.0f}"]
    if vol_declining:
        reasons.append("Volume declining in consolidation")
    if last["close"] > last["vwap"]:
        reasons.append("Price above VWAP")

    return {"setup": "bull_flag", "direction": "call", "entry": entry,
            "stop": stop, "target": target, "confidence": confidence,
            "timeframe": "weekly", "signals": reasons}


def detect_vwap_breakdown(df):
    if df is None or len(df) < 20:
        return None
    df = df.copy()
    df["vwap"] = _vwap(df)
    df["rsi"] = _rsi(df["close"])
    last = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_below = prev["close"] >= prev["vwap"] and last["close"] < last["vwap"]
    if not crossed_below:
        return None
    rsi_val = last["rsi"]
    if not (28 <= rsi_val <= 52):
        return None

    avg_vol = df.tail(20)["volume"].mean()
    vol_expand = last["volume"] > avg_vol * 1.2

    confidence = 0.60
    confidence += 0.10 if vol_expand else 0
    confidence += 0.06 if rsi_val < 45 else 0

    entry = round(last["close"] * 0.999, 2)
    stop = round(last["vwap"] * 1.003, 2)
    risk = stop - entry
    target = round(entry - risk * 2.2, 2)

    reasons = ["VWAP breakdown", f"RSI {rsi_val:.0f}"]
    if vol_expand:
        reasons.append("Volume expansion on break")

    return {"setup": "vwap_breakdown", "direction": "put", "entry": entry,
            "stop": stop, "target": target, "confidence": confidence,
            "timeframe": "weekly", "signals": reasons}


def detect_iv_squeeze(df, finnhub_indicator=None):
    if df is None or len(df) < 40:
        return None
    bb_pct = _bb_width_pct(df["close"]).iloc[-1]
    if bb_pct > 0.20:
        return None

    direction = "call"
    if finnhub_indicator:
        sig = finnhub_indicator.get("signal", "neutral")
        if sig == "sell":
            direction = "put"
        elif sig not in ("buy", "sell"):
            return None

    confidence = min(0.65 + (0.20 - bb_pct) * 0.5, 0.88)
    last = df.iloc[-1]
    atr_v = _atr(df).iloc[-1]
    entry = round(last["close"] * (1.001 if direction == "call" else 0.999), 2)
    stop = round(entry - atr_v * 1.5 if direction == "call" else entry + atr_v * 1.5, 2)
    risk = abs(entry - stop)
    target = round(entry + risk * 2.5 if direction == "call" else entry - risk * 2.5, 2)

    return {"setup": "iv_squeeze", "direction": direction, "entry": entry,
            "stop": stop, "target": target, "confidence": confidence,
            "timeframe": "swing", "signals": [
                f"BB squeeze pct={bb_pct:.2f}",
                f"Aggregate: {direction.upper()}"
            ]}


def score_news_sentiment(news_items):
    if not news_items:
        return 0.0
    bullish_kw = {"upgrade", "buy", "beat", "surge", "record", "rally", "bullish"}
    bearish_kw = {"downgrade", "sell", "miss", "drop", "warning", "bearish", "loss"}
    scores = []
    for item in news_items[:20]:
        text = (item.get("headline", "") + " " + item.get("summary", "")).lower()
        b = sum(1 for w in bullish_kw if w in text)
        s = sum(1 for w in bearish_kw if w in text)
        t = b + s
        if t:
            scores.append((b - s) / t)
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def evaluate_symbol(symbol, candles_5m, candles_1h, quote, news,
                    finnhub_indicator=None, vix=None):
    if candles_5m is None and candles_1h is None:
        return None

    df = candles_5m if candles_5m is not None else candles_1h

    if is_chop_day(candles_5m, vix):
        log.info(f"{symbol}: chop day — skipping")
        return None

    sentiment = score_news_sentiment(news)
    candidates = []

    for detector in [detect_bull_flag, detect_vwap_breakdown]:
        result = detector(df)
        if result:
            candidates.append(result)

    squeeze = detect_iv_squeeze(df, finnhub_indicator)
    if squeeze:
        candidates.append(squeeze)

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x["confidence"])

    # Sentiment boost/penalty
    if best["direction"] == "call" and sentiment > 0.2:
        best["confidence"] = min(best["confidence"] + 0.05, 0.95)
        best["signals"].append(f"Bullish news ({sentiment:+.2f})")
    elif best["direction"] == "put" and sentiment < -0.2:
        best["confidence"] = min(best["confidence"] + 0.05, 0.95)
        best["signals"].append(f"Bearish news ({sentiment:+.2f})")
    elif best["direction"] == "call" and sentiment < -0.3:
        best["confidence"] *= 0.85
    elif best["direction"] == "put" and sentiment > 0.3:
        best["confidence"] *= 0.85

    risk = abs(best["entry"] - best["stop"])
    reward = abs(best["target"] - best["entry"])
    rr = reward / risk if risk > 0 else 0

    if best["confidence"] < MIN_CONFIDENCE or rr < MIN_RR:
        return None

    return TradeAlert(
        symbol=symbol,
        setup=best["setup"],
        direction=best["direction"],
        entry=best["entry"],
        stop=best["stop"],
        target=best["target"],
        confidence=round(best["confidence"], 3),
        risk_reward=round(rr, 2),
        timeframe=best.get("timeframe", "weekly"),
        signals=best["signals"],
    )
