"""
agents/sentiment.py
───────────────────
SentimentAgent — news headline scoring + price trend alignment.

Uses:
  • Finnhub news items (already fetched by run_scan)
  • Daily candles for 5-day price trend
  • Quote for current momentum context
"""
import logging
from datetime import datetime, timezone, timedelta

from agents.base import AgentView

log = logging.getLogger(__name__)

BULLISH_KW = {"upgrade","buy","beat","surge","record","rally","bullish","strong",
              "outperform","positive","growth","profit","gain","breakout","top"}
BEARISH_KW = {"downgrade","sell","miss","drop","warning","bearish","loss","weak",
              "disappoint","concern","risk","decline","cut","probe","lawsuit","fraud"}


class SentimentAgent:
    name = "sentiment"

    def analyze(
        self,
        symbol: str,
        news: list[dict],
        quote: "dict | None",
        candles_1d: "pd.DataFrame | None",
    ) -> AgentView:
        try:
            return self._run(symbol, news, quote, candles_1d)
        except Exception as e:
            log.error(f"SentimentAgent {symbol}: {e}")
            return AgentView(agent=self.name, symbol=symbol, direction="neutral",
                             confidence=0.40, reasons=[f"Analysis error: {e}"])

    def _run(self, symbol, news, quote, candles_1d) -> AgentView:
        # ── News scoring ──────────────────────────────────────────
        news_score, recent_count, article_detail = self._score_news(news)

        # ── Price trend (5-day slope) ─────────────────────────────
        price_trend = self._price_trend(candles_1d)

        # ── Direction determination ───────────────────────────────
        news_bullish  = news_score > 0.10
        news_bearish  = news_score < -0.10
        trend_bullish = price_trend > 0
        trend_bearish = price_trend < 0

        if news_bullish and trend_bullish:
            direction = "call"
            aligned   = True
        elif news_bearish and trend_bearish:
            direction = "put"
            aligned   = True
        elif news_bullish:
            direction = "call"
            aligned   = False
        elif news_bearish:
            direction = "put"
            aligned   = False
        else:
            direction = "neutral"
            aligned   = False

        # ── Confidence ────────────────────────────────────────────
        conf = 0.45 + abs(news_score) * 0.35
        if aligned:
            conf += 0.10
        if recent_count >= 3:
            conf += 0.05    # high coverage = more signal
        conf = min(conf, 0.85)

        # ── Reasons ───────────────────────────────────────────────
        reasons = []
        if news_score >= 0.1:
            reasons.append(f"Bullish news sentiment ({news_score:+.2f})")
        elif news_score <= -0.1:
            reasons.append(f"Bearish news sentiment ({news_score:+.2f})")
        else:
            reasons.append(f"Neutral news ({news_score:+.2f})")
        reasons.append(f"{recent_count} articles in 24h")
        if price_trend != 0:
            reasons.append(f"5-day price trend: {'rising' if price_trend > 0 else 'falling'} ({price_trend:+.2f}%)")
        if aligned:
            reasons.append("News and price trend aligned")

        return AgentView(
            agent=self.name, symbol=symbol, direction=direction,
            confidence=round(conf, 3), reasons=reasons,
            data={
                "news_score":      round(news_score, 3),
                "headline_count":  recent_count,
                "price_trend_5d":  round(price_trend, 2),
                "aligned":         aligned,
                "top_headlines":   article_detail[:3],
            },
        )

    def _score_news(self, news: list[dict]) -> tuple[float, int, list[str]]:
        if not news:
            return 0.0, 0, []
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=24)
        scores  = []
        recent  = 0
        details = []

        for item in news[:30]:
            text = (item.get("headline", "") + " " + item.get("summary", "")).lower()
            b = sum(1 for w in BULLISH_KW if w in text)
            s = sum(1 for w in BEARISH_KW if w in text)
            t = b + s
            if t:
                scores.append((b - s) / t)

            # Count recent articles
            ts = item.get("datetime", 0)
            try:
                pub = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
                if pub and pub >= cutoff:
                    recent += 1
            except Exception:
                pass

            if item.get("headline"):
                details.append(item["headline"][:80])

        score = round(sum(scores) / len(scores), 3) if scores else 0.0
        return score, recent, details

    def _price_trend(self, candles_1d) -> float:
        """Return 5-day percentage price change. 0 if no data."""
        if candles_1d is None or len(candles_1d) < 6:
            return 0.0
        closes = candles_1d["close"].tail(6)
        pct = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100
        return round(float(pct), 2)
