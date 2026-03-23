"""
data/market.py
──────────────
Market data via yfinance — free, no API key, no rate limits.
Finnhub is used only for news (see data/finnhub_client.py).
"""
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Watchlist — high-beta, options-liquid, squeeze-prone symbols ──────────
# Curated via walk-forward backtest (60d, Jan-Mar 2026):
#   Proven winners kept; mega-caps, semis, ETFs removed (poor WR).
#   Expanded with similar high-beta names in EV, crypto, fintech, consumer.
WATCHLIST = [
    # Curated 15 — high-beta, high-options-volume, proven movers.
    # Criteria: daily options volume >10k contracts, beta >1.5, squeeze-prone.
    "TSLA",   # highest retail options volume, extreme beta
    "COIN",   # crypto proxy, massive IV swings
    "NVDA",   # AI momentum anchor, deep options chain
    "META",   # high-beta mega-cap, reliable setups
    "MARA",   # BTC miner, extreme beta (2-5x BTC moves)
    "RIOT",   # BTC miner, high squeeze potential
    "RBLX",   # growth proxy, strong options flow
    "LCID",   # EV, extreme beta, high short interest
    "RIVN",   # EV, high-beta, squeeze history
    "SMCI",   # AI infrastructure, volatile
    "AFRM",   # fintech, extreme beta (earnings swings 20-40%)
    "GME",    # meme/squeeze, very high options activity
    "DKNG",   # high-beta consumer speculative
    "HOOD",   # crypto/retail broker, high beta
    "PLTR",   # AI/defense, high retail interest and options volume
]

# ── S&P 500 symbol cache (populated lazily) ──
_sp500_cache: list[str] = []

def get_sp500_symbols() -> list[str]:
    """
    Fetch current S&P 500 constituents from Wikipedia.
    Falls back to WATCHLIST if unavailable.
    Result is cached for the process lifetime.
    """
    global _sp500_cache
    if _sp500_cache:
        return _sp500_cache
    import io, requests as _req
    try:
        resp = _req.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (compatible; TradeAgent/1.0)"},
            timeout=10,
        )
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
        symbols = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        _sp500_cache = sorted(symbols)
        log.info(f"S&P 500 loaded: {len(_sp500_cache)} symbols")
        return _sp500_cache
    except Exception as e:
        log.warning(f"S&P 500 fetch failed, falling back to WATCHLIST: {e}")
        return WATCHLIST

log = logging.getLogger(__name__)

def get_candles(symbol: str, interval: str = "5m", days: int = 3) -> pd.DataFrame | None:
    """
    Fetch OHLCV candles via yfinance — free, no API key, no rate limit.

    interval options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d
    Max history:      1m=7days, 5m/15m=60days, 1h=730days, 1d=unlimited

    Returns DataFrame with columns: open, high, low, close, volume, datetime
    """
    try:
        period = f"{days}d"
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or df.empty:
            log.warning(f"{symbol}: no candle data returned")
            return None
        df = df.rename(columns=str.lower)
        df = df[["open", "high", "low", "close", "volume"]]
        df.index.name = "datetime"
        df = df.reset_index()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception as e:
        log.error(f"{symbol} candle fetch error: {e}")
        return None

def get_quote(symbol: str) -> dict | None:
    """
    Fast quote via yfinance — price, change%, volume.
    """
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info
        price = getattr(info, "last_price", None)
        prev  = getattr(info, "previous_close", None)
        if not price:
            return None
        change_pct = round((price - prev) / prev * 100, 2) if prev else 0
        return {
            "symbol":     symbol,
            "c":          round(price, 2),
            "pc":         round(prev, 2) if prev else None,
            "change_pct": change_pct,
            "h":          getattr(info, "day_high", None),
            "l":          getattr(info, "day_low", None),
            "v":          getattr(info, "three_month_average_volume", None),
        }
    except Exception as e:
        log.error(f"{symbol} quote error: {e}")
        return None


def batch_prescreen(symbols: list[str], top_n: int = 75) -> list[str]:
    """
    Batch pre-screen symbols using 20-day daily candles downloaded in one call.
    Scores each symbol by 5-day absolute return × volume ratio, returns top_n
    sorted by score descending.  Falls back to symbols[:top_n] on any error.
    """
    try:
        # Batch in chunks of 100 to avoid thread exhaustion / DNS failures
        CHUNK = 100
        frames = []
        for i in range(0, len(symbols), CHUNK):
            chunk = symbols[i:i+CHUNK]
            tickers = " ".join(chunk)
            try:
                part = yf.download(
                    tickers,
                    period="20d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=False,   # avoid getaddrinfo thread exhaustion
                )
                frames.append((chunk, part))
            except Exception as _ce:
                log.warning(f"batch_prescreen chunk {i//CHUNK} failed: {_ce}")
                continue
        if not frames:
            return symbols[:top_n]

        scores: dict[str, float] = {}
        for chunk, df in frames:
            for sym in chunk:
                try:
                    # Handle both multi-ticker (df[sym]['Close']) and
                    # single-ticker (df['Close']) DataFrame formats.
                    if len(chunk) == 1:
                        close  = df["Close"].dropna()
                        volume = df["Volume"].dropna()
                    else:
                        close  = df[sym]["Close"].dropna()
                        volume = df[sym]["Volume"].dropna()

                    if len(close) < 5 or len(volume) < 5:
                        continue

                    ret_5d    = abs(float(close.iloc[-1]) / float(close.iloc[-5]) - 1)
                    vol_ratio = float(volume.iloc[-1]) / max(float(volume.iloc[-5:].mean()), 1)
                    vol_ratio = max(0.0, min(3.0, vol_ratio))
                    scores[sym] = ret_5d * max(vol_ratio, 1.0)
                except Exception:
                    continue

        ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
        return ranked[:top_n] if ranked else symbols[:top_n]
    except Exception as e:
        log.warning(f"batch_prescreen failed, returning first {top_n} symbols: {e}")
        return symbols[:top_n]


def get_scan_universe() -> list[str]:
    """
    Return the full S&P 500 universe (used by main.py for expanded scanning).
    """
    return get_sp500_symbols()
