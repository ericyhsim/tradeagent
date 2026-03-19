"""
data/finnhub_client.py
──────────────────────
Finnhub integration — news only.

All price data (quotes, candles) is sourced from yfinance (data/market.py).
Finnhub is kept solely for company news because it's reliable on the free tier.
"""

import os, time, logging
from datetime import datetime, timedelta
from collections import deque
from typing import Optional
import threading
import finnhub

log = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket — stays under Finnhub free tier limit of 60 calls/min."""
    def __init__(self, calls_per_minute: int = 55):
        self._calls = calls_per_minute
        self._timestamps: deque = deque()
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.time()
            while self._timestamps and now - self._timestamps[0] > 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self._calls:
                sleep_for = 60 - (now - self._timestamps[0]) + 0.05
                time.sleep(sleep_for)
            self._timestamps.append(time.time())


class FinnhubNewsClient:
    """
    Thin Finnhub wrapper — company news only.

    Usage:
        client = FinnhubNewsClient(api_key=os.environ["FINNHUB_API_KEY"])
        news = client.get_news("NVDA", days=1)
    """

    def __init__(self, api_key: str):
        self._client = finnhub.Client(api_key=api_key)
        self._rl = RateLimiter(calls_per_minute=55)

    def _call(self, fn, *args, **kwargs):
        self._rl.wait()
        try:
            return fn(*args, **kwargs)
        except finnhub.FinnhubAPIException as e:
            log.error(f"Finnhub API error: {e}")
            return None
        except Exception as e:
            log.error(f"Finnhub unexpected error: {e}")
            return None

    def get_news(self, symbol: str, days: int = 1) -> list[dict]:
        """Company news articles for a symbol."""
        to_date   = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        res = self._call(self._client.company_news, symbol, _from=from_date, to=to_date)
        return res or []
