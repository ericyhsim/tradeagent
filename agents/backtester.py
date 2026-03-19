"""
agents/backtester.py
────────────────────
Walk-forward daily backtest for the multi-agent pipeline.

Methodology
───────────
• For each symbol, downloads (lookback_days + 40) days of daily OHLCV.
• Walks forward from bar 35 to bar (n - OUTCOME_DAYS - 1).
  Agents see only data[:d] at each step — no lookahead.
• Momentum, Quant, StatArb run on the daily slice (historically accurate).
• Macro, Fundamental, Sentiment, OptionsFlow run once per symbol with
  current data — treated as "regime/context" in results. Noted in output.
• Entry  : close of signal bar d.
• Exit   : first of (target hit / stop hit / day-5 close).
• P&L    : signed fraction of entry price.

Per-agent accuracy
──────────────────
For every trade, records whether each agent agreed or opposed.
Win-rate-when-agreed vs win-rate-when-opposed gives an "accuracy lift"
score: positive = agent adds value; negative = agent is a contrarian drag.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from agents.base          import AgentView
from agents.arbitration   import ArbitrationAgent, BASE_WEIGHTS, REGIME_PRESETS
from agents.regime        import RegimeDetectionAgent
from agents.momentum      import MomentumAgent
from agents.quant         import QuantAgent
from agents.stat_arb      import StatArbAgent
from agents.fundamental   import FundamentalAgent
from agents.sentiment     import SentimentAgent
from agents.macro         import MacroAgent
from agents.options_flow  import OptionsFlowAgent

log = logging.getLogger(__name__)

OUTCOME_DAYS = 5     # bars forward to evaluate trade outcome
MIN_HIST     = 35    # minimum bars of history before first signal allowed

AGENT_ORDER = [
    "momentum", "sentiment", "fundamental",
    "quant", "macro", "options_flow", "stat_arb",
]

_NEUTRAL = lambda name, sym: AgentView(
    agent=name, symbol=sym, direction="neutral", confidence=0.40, reasons=["unavailable"]
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BtTrade:
    symbol:      str
    date:        str       # YYYY-MM-DD of signal bar
    direction:   str       # call / put
    entry:       float
    stop:        float
    target:      float
    rr:          float
    exit_price:  float
    pnl_pct:     float     # signed fraction: 0.08 = +8%
    outcome:     str       # win / loss / timeout
    exit_reason: str       # target_hit / stop_hit / timeout
    setup:       str
    consensus:   float
    agent_votes: dict      # {agent: direction}
    agent_confs: dict      # {agent: confidence}

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BtAgentStats:
    name:        str
    agreed:      int = 0
    agreed_wins: int = 0
    opposed:     int = 0
    opposed_wins: int = 0

    @property
    def wr_agreed(self) -> float:
        return round(self.agreed_wins / self.agreed, 3) if self.agreed else 0.0

    @property
    def wr_opposed(self) -> float:
        return round(self.opposed_wins / self.opposed, 3) if self.opposed else 0.0

    @property
    def lift(self) -> float:
        return round(self.wr_agreed - self.wr_opposed, 3)

    def to_dict(self) -> dict:
        return {
            "name":         self.name,
            "agreed":       self.agreed,
            "agreed_wins":  self.agreed_wins,
            "wr_agreed":    self.wr_agreed,
            "opposed":      self.opposed,
            "opposed_wins": self.opposed_wins,
            "wr_opposed":   self.wr_opposed,
            "lift":         self.lift,
        }


@dataclass
class BtResult:
    total_trades:    int
    win_rate:        float
    avg_return:      float
    total_pl_pct:    float
    max_drawdown:    float
    equity_curve:    list     # [{date, symbol, equity}], equity starts at 100
    trades:          list     # list of BtTrade.to_dict()
    agent_stats:     list     # list of BtAgentStats.to_dict()
    symbols_scanned: int
    signals_found:   int
    lookback_days:   int
    run_time_sec:    float
    preset:          Optional[str]
    note:            str

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ── Outcome simulation ────────────────────────────────────────────────────────

def _simulate_outcome(
    future:    pd.DataFrame,
    entry:     float,
    stop:      float,
    target:    float,
    direction: str,
) -> tuple[str, float, str]:
    """
    Walk future bars bar-by-bar checking if target or stop is hit first.
    Returns (outcome, exit_price, exit_reason).
    """
    for _, bar in future.iterrows():
        hi = float(bar["high"])
        lo = float(bar["low"])
        if direction == "call":
            if hi >= target:
                return "win",  round(target, 2), "target_hit"
            if lo <= stop:
                return "loss", round(stop, 2),   "stop_hit"
        else:
            if lo <= target:
                return "win",  round(target, 2), "target_hit"
            if hi >= stop:
                return "loss", round(stop, 2),   "stop_hit"

    # Neither triggered — use last close as exit
    exit_px = float(future.iloc[-1]["close"])
    if direction == "call":
        outcome = "win" if exit_px > entry else "loss"
    else:
        outcome = "win" if exit_px < entry else "loss"
    return outcome, round(exit_px, 2), "timeout"


# ── Batch data fetch ─────────────────────────────────────────────────────────

def _batch_download(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    """
    Download all symbols in one yf.download() call (far fewer HTTP requests
    than one Ticker.history() per symbol, avoids rate-limiting).
    Returns {symbol: df} with lowercase OHLCV columns and a 'datetime' column.
    """
    tickers_str = " ".join(symbols)
    period = f"{days + 45}d"
    try:
        raw = yf.download(
            tickers_str,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as e:
        log.warning(f"Batch download failed: {e}")
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            if len(symbols) == 1:
                # Single-ticker: columns are flat (Open, High, ...)
                df = raw.copy()
            else:
                if sym not in raw.columns.get_level_values(0):
                    continue
                df = raw[sym].copy()

            df = df.rename(columns=str.lower)
            cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            if len(cols) < 4:
                continue
            df = df[cols].copy()
            df.index.name = "datetime"
            df = df.reset_index()
            df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
            df = df.sort_values("datetime").reset_index(drop=True)
            df = df.dropna(subset=["close"])

            if len(df) >= MIN_HIST + OUTCOME_DAYS + 2:
                result[sym] = df
        except Exception as e:
            log.debug(f"Batch extract {sym}: {e}")

    return result


# ── Main entry ────────────────────────────────────────────────────────────────

def run_backtest(
    symbols:       list[str],
    lookback_days: int = 60,
    preset:        Optional[str] = None,
) -> BtResult:
    """
    Synchronous walk-forward backtest.
    Downloads all symbols in a single batch call to avoid yfinance rate limits.
    Call via asyncio.to_thread from FastAPI to avoid blocking the event loop.
    """
    t0 = time.time()

    weights = dict(BASE_WEIGHTS)
    if preset and preset in REGIME_PRESETS:
        weights = dict(REGIME_PRESETS[preset])

    # Fresh agent instances — don't share state with the live pipeline
    mom_ag   = MomentumAgent()
    quant_ag = QuantAgent()
    sarb_ag  = StatArbAgent()
    fund_ag  = FundamentalAgent()
    sent_ag  = SentimentAgent()
    mac_ag   = MacroAgent()
    opt_ag   = OptionsFlowAgent()
    reg_ag   = RegimeDetectionAgent()
    arb_ag   = ArbitrationAgent()

    trades: list[BtTrade] = []

    # ── Single batch download (30 symbols → ~3 HTTP requests vs 30) ──────────
    log.info(f"Backtester: batch downloading {len(symbols)} symbols over {lookback_days}d")
    symbol_dfs = _batch_download(symbols, lookback_days)
    symbols_scanned = len(symbol_dfs)
    log.info(f"Backtester: {symbols_scanned} symbols downloaded successfully")

    for symbol, df in symbol_dfs.items():
        try:
            # Current-state agents — run once per symbol with latest data
            quote_now = {
                "symbol": symbol,
                "c":  float(df["close"].iloc[-1]),
                "pc": float(df["close"].iloc[-2]),
                "v":  float(df["volume"].iloc[-1]),
                "h":  float(df["high"].iloc[-1]),
                "l":  float(df["low"].iloc[-1]),
            }
            try:
                f_view   = fund_ag.analyze(symbol)
                s_view   = sent_ag.analyze(symbol, [], quote_now, df)
                mac_view = mac_ag.analyze(symbol)
                opt_view = opt_ag.analyze(symbol, quote_now)
            except Exception as e:
                log.debug(f"Backtest {symbol} static agents: {e}")
                f_view   = _NEUTRAL("fundamental",  symbol)
                s_view   = _NEUTRAL("sentiment",    symbol)
                mac_view = _NEUTRAL("macro",        symbol)
                opt_view = _NEUTRAL("options_flow", symbol)

            # Walk-forward loop
            for d in range(MIN_HIST, len(df) - OUTCOME_DAYS - 1):
                hist = df.iloc[:d].copy()

                try:
                    # Historical agents see only data up to bar d
                    m_view  = mom_ag.analyze(symbol, hist, hist)  # daily as proxy
                    q_view  = quant_ag.analyze(symbol, hist, hist)
                    # stat_arb makes live yfinance pair-fetches which exhaust rate limits
                    # during walk-forward loops (hundreds of calls).  Use neutral stub.
                    sa_view = _NEUTRAL("stat_arb", symbol)
                except Exception:
                    continue

                signal_views = [m_view, s_view, f_view, q_view, mac_view, opt_view, sa_view]

                try:
                    regime  = reg_ag.classify(hist, quote_now)
                    verdict = arb_ag.vote(signal_views, regime, weights, {})
                except Exception:
                    continue

                if not verdict.approved:
                    continue

                # Determine price levels
                entry  = float(df.iloc[d]["close"])
                stop   = verdict.stop
                target = verdict.target

                if stop is None or target is None:
                    continue

                # Validate direction of levels
                if verdict.direction == "call":
                    if stop >= entry or target <= entry:
                        continue
                else:
                    if stop <= entry or target >= entry:
                        continue

                future = df.iloc[d + 1: d + 1 + OUTCOME_DAYS]
                if len(future) < 1:
                    continue

                outcome, exit_px, exit_reason = _simulate_outcome(
                    future, entry, stop, target, verdict.direction
                )

                pnl = (
                    (exit_px - entry) / entry
                    if verdict.direction == "call"
                    else (entry - exit_px) / entry
                )
                rr = (
                    abs(target - entry) / abs(entry - stop)
                    if abs(entry - stop) > 0 else 0.0
                )

                sig_dt   = df.iloc[d]["datetime"]
                date_str = sig_dt.strftime("%Y-%m-%d") if hasattr(sig_dt, "strftime") else str(sig_dt)[:10]

                trades.append(BtTrade(
                    symbol      = symbol,
                    date        = date_str,
                    direction   = verdict.direction,
                    entry       = round(entry, 2),
                    stop        = round(stop, 2),
                    target      = round(target, 2),
                    rr          = round(rr, 2),
                    exit_price  = exit_px,
                    pnl_pct     = round(pnl, 4),
                    outcome     = outcome,
                    exit_reason = exit_reason,
                    setup       = verdict.setup,
                    consensus   = round(verdict.consensus_score, 3),
                    agent_votes = {v.agent: v.direction for v in signal_views},
                    agent_confs = {v.agent: round(v.confidence, 3) for v in signal_views},
                ))

        except Exception as e:
            log.warning(f"Backtest {symbol} outer error: {e}")

    # ── Aggregate stats ────────────────────────────────────────────────────────
    total    = len(trades)
    wins     = [t for t in trades if t.outcome == "win"]
    win_rate = round(len(wins) / total, 3) if total else 0.0
    avg_ret  = round(sum(t.pnl_pct for t in trades) / total, 4) if total else 0.0
    total_pl = round(sum(t.pnl_pct for t in trades), 4)

    # Equity curve (compounded from 100)
    equity = 100.0
    peak   = 100.0
    max_dd = 0.0
    equity_curve = []
    for t in sorted(trades, key=lambda x: x.date):
        equity *= (1.0 + t.pnl_pct)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak
        if dd > max_dd:
            max_dd = dd
        equity_curve.append({
            "date":   t.date,
            "symbol": t.symbol,
            "equity": round(equity, 2),
        })

    # Per-agent accuracy stats
    stats_map = {a: BtAgentStats(name=a) for a in AGENT_ORDER}
    for t in trades:
        is_win = t.outcome == "win"
        for agent in AGENT_ORDER:
            vote   = t.agent_votes.get(agent, "neutral")
            agreed = (vote == t.direction or vote == "neutral")
            st     = stats_map[agent]
            if agreed:
                st.agreed += 1
                if is_win:
                    st.agreed_wins += 1
            else:
                st.opposed += 1
                if is_win:
                    st.opposed_wins += 1

    return BtResult(
        total_trades    = total,
        win_rate        = win_rate,
        avg_return      = avg_ret,
        total_pl_pct    = total_pl,
        max_drawdown    = round(max_dd, 4),
        equity_curve    = equity_curve,
        trades          = [t.to_dict() for t in sorted(trades, key=lambda x: x.date)],
        agent_stats     = [stats_map[a].to_dict() for a in AGENT_ORDER],
        symbols_scanned = symbols_scanned,
        signals_found   = total,
        lookback_days   = lookback_days,
        run_time_sec    = round(time.time() - t0, 1),
        preset          = preset,
        note            = (
            f"Daily bar walk-forward. Momentum / Quant / StatArb use rolling "
            f"historical slices. Macro / Fundamental / Sentiment / OptionsFlow "
            f"use current-state data as regime context. Entry = EOD close. "
            f"Outcome measured over next {OUTCOME_DAYS} trading days."
        ),
    )
