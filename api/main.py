"""
api/main.py  —  TradeAgent backend

Data:    yfinance (quotes/candles)  |  Finnhub (news only)
Trading: Tradier sandbox (equity + options paper trading)

Endpoints
─────────
GET  /                              dashboard
GET  /api/status
GET  /api/signals                   latest alerts
GET  /api/quotes                    all cached quotes
GET  /api/quote/{symbol}            single live quote
GET  /api/scan                      trigger manual scan
GET  /api/equity                    equity balance history
GET  /api/watchlist
GET  /api/news/{symbol}
GET  /api/logs
GET  /api/balances                  Tradier account balances
GET  /api/positions                 open positions + live P&L
GET  /api/orders                    order history
POST /api/orders                    place equity order
POST /api/positions/{symbol}/close  close a position (equity or option)
DELETE /api/orders/{order_id}       cancel order
GET  /api/options/expirations/{symbol}
GET  /api/options/chain/{symbol}?expiration=YYYY-MM-DD
POST /api/options/orders            place option order
WS   /ws/live                       real-time stream
"""

import os, asyncio, logging, re, json, csv, io
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, date
from pathlib import Path
from collections import deque
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=_env_path, override=True)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from data.market import get_candles, get_quote, WATCHLIST
from data.finnhub_client import FinnhubNewsClient
from data.tradier_client import TradierClient, is_option_symbol
from signals.engine import evaluate_symbol
from agents.feedback     import FeedbackEngine
from agents.orchestrator import Orchestrator
from agents.exit_manager  import ExitManager, ScaleTargets
import agents.orchestrator  as _orch_module
import agents.arbitration   as _arb_module   # BASE_WEIGHTS lives here

ET = ZoneInfo("America/New_York")

# ── Config ───────────────────────────────────────

MAX_RISK_PER_TRADE = float(os.environ.get("MAX_RISK_PER_TRADE", "0.02"))
MAX_DAILY_LOSS     = float(os.environ.get("MAX_DAILY_LOSS", "0.03"))
AUTO_TRADE         = os.environ.get("AUTO_TRADE", "true").lower() != "false"

# ── Clients ──────────────────────────────────────

_fh_key = os.environ.get("FINNHUB_API_KEY", "")
fh: Optional[FinnhubNewsClient] = FinnhubNewsClient(_fh_key) if _fh_key else None

_tradier_token   = os.environ.get("TRADIER_ACCESS_TOKEN", "")
_tradier_account = os.environ.get("TRADIER_ACCOUNT_ID", "")
_tradier_paper   = os.environ.get("TRADIER_PAPER", "true").lower() != "false"
tradier: Optional[TradierClient] = (
    TradierClient(_tradier_token, _tradier_account, paper=_tradier_paper)
    if _tradier_token and _tradier_account else None
)
if tradier:
    log.info(f"Tradier connected — {'paper sandbox' if _tradier_paper else 'LIVE'} | auto_trade={AUTO_TRADE}")

# ── Multi-agent system ────────────────────────────

feedback     = FeedbackEngine()
orchestrator = Orchestrator(feedback_engine=feedback)
exit_manager = ExitManager()
log.info("Multi-agent orchestrator initialized")

# ── State ────────────────────────────────────────

state = {
    "alerts":         [],
    "quotes":         {},
    "equity_history": [],
    "traded_today":   set(),
    "daily_pl":       0.0,
    "opening_equity": None,
    "scan_running":   False,
    "last_scan":      None,
    "log_entries":    deque(maxlen=500),
    "connected_ws":   set(),
    "signal_log":     {},   # symbol → {signal_id, alert_dict} for feedback tracking
    "agent_views":    {},   # symbol → latest per-agent views (for UI)
    "scale_targets":  {},   # option_symbol → ScaleTargets (for tiered exit monitoring)
    "pending_closes": {},   # option_symbol → {order_id, qty, type, placed_at}
}

# ── Signal history persistence ────────────────────
# Signals are grouped by trading date and stored in logs/signal_history.json.
# We keep the last 10 distinct trading session dates.

_LOGS_DIR          = Path(__file__).parent.parent / "logs"
_SIGNAL_HISTORY_FILE = _LOGS_DIR / "signal_history.json"

def _load_signal_history() -> list[dict]:
    """Load persisted signal history from disk; returns [] on any error."""
    try:
        if _SIGNAL_HISTORY_FILE.exists():
            data = json.loads(_SIGNAL_HISTORY_FILE.read_text())
            return data if isinstance(data, list) else []
    except Exception as e:
        log.warning(f"Could not load signal history: {e}")
    return []

def _save_signal_history(history: list[dict]) -> None:
    try:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _SIGNAL_HISTORY_FILE.write_text(json.dumps(history, default=str))
    except Exception as e:
        log.warning(f"Could not save signal history: {e}")

def _append_to_history(alert_dict: dict) -> None:
    """
    Add a signal to the persistent history, tagged with today's date.
    Keeps only the last 10 distinct trading session dates.
    """
    today = date.today().isoformat()
    history: list[dict] = _load_signal_history()

    entry = {**alert_dict, "session_date": today}
    history.append(entry)

    # Deduplicate by signal_id to avoid double-counting on server restart
    seen_ids: set = set()
    unique = []
    for h in history:
        sid = h.get("signal_id") or id(h)
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique.append(h)

    # Keep only signals from the last 10 distinct session dates
    all_dates = sorted({h["session_date"] for h in unique}, reverse=True)
    keep_dates = set(all_dates[:10])
    unique = [h for h in unique if h.get("session_date") in keep_dates]

    _save_signal_history(unique)


def add_log(agent: str, message: str, level: str = "info"):
    entry = {"time": datetime.now().strftime("%H:%M:%S"), "agent": agent,
             "message": message, "level": level}
    state["log_entries"].appendleft(entry)
    log.info(f"[{agent}] {message}")


async def broadcast(payload: dict):
    dead = set()
    for ws in state["connected_ws"]:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    state["connected_ws"] -= dead


def is_market_open() -> bool:
    now = datetime.now(ET)
    if now.weekday() >= 5:
        return False
    mins = now.hour * 60 + now.minute
    return 570 <= mins <= 960

# ── Balance snapshot ─────────────────────────────

async def snapshot_balance():
    if not tradier:
        return
    balances = await asyncio.to_thread(tradier.get_balances)
    if not balances:
        return
    equity = balances.get("total_equity", 0)
    cash   = balances.get("total_cash", 0)
    now    = datetime.now()
    if state["opening_equity"] is None:
        state["opening_equity"] = equity
    state["daily_pl"] = round(equity - state["opening_equity"], 2)
    state["equity_history"].append({
        "time":  now.strftime("%H:%M"),
        "date":  now.strftime("%b %d"),
        "value": round(equity, 2),
        "cash":  round(cash, 2),
    })
    await broadcast({"type": "balance", "equity": round(equity, 2),
                     "cash": round(cash, 2), "daily_pl": state["daily_pl"]})


# ── Options auto-trade helper ─────────────────────

async def _find_option_and_trade(symbol: str, direction: str, entry: float,
                                  risk_dollars: float, alert: dict) -> bool:
    """
    Look up the options chain, find nearest ATM contract, place buy_to_open.
    Returns True if an order was placed.
    """
    expirations = await asyncio.to_thread(tradier.get_option_expirations, symbol)
    if not expirations:
        return False

    today_date = date.today()
    target_exp = None
    for exp_str in sorted(expirations):
        try:
            exp_date = date.fromisoformat(exp_str)
        except ValueError:
            continue
        days_out = (exp_date - today_date).days
        if 5 <= days_out <= 30:
            target_exp = exp_str
            break

    if not target_exp:
        return False

    chain = await asyncio.to_thread(tradier.get_option_chain, symbol, target_exp)
    opt_type  = "call" if direction == "call" else "put"
    contracts = [o for o in chain
                 if o.get("option_type") == opt_type and (o.get("ask") or 0) > 0]
    if not contracts:
        return False

    # Pick strike closest to entry (ATM)
    contracts.sort(key=lambda o: abs((o.get("strike") or 0) - entry))
    contract    = contracts[0]
    ask         = contract.get("ask", 0)
    opt_symbol  = contract.get("symbol", "")
    strike      = contract.get("strike", 0)
    delta       = round(contract.get("greeks", {}).get("delta", 0) or 0, 2)

    if not opt_symbol or ask <= 0:
        return False

    bid = contract.get("bid", 0) or 0
    mid = round((bid + ask) / 2, 2) if bid else ask

    # Limit at the bid — fills near the bottom of the spread.
    # If there's no bid (illiquid), fall back to 5% below ask.
    # Missing a fill is fine; getting a bad fill at ask+2% is not.
    limit_price       = round(bid if bid >= 0.05 else mid * 0.95, 2)
    limit_price       = max(limit_price, 0.05)  # Tradier minimum

    cost_per_contract = mid * 100   # size based on mid, not the limit price
    num_contracts     = max(1, int(risk_dollars / cost_per_contract))
    num_contracts     = min(num_contracts, 10)

    result = await asyncio.to_thread(
        tradier.place_option_order,
        opt_symbol, "buy_to_open", num_contracts, "limit", limit_price, "gtc",
    )

    if result and result.get("order", {}).get("status") == "ok":
        order_id = result["order"]["id"]
        add_log("Execution",
            f"Options BTO: {num_contracts}x {opt_symbol} lmt ${limit_price:.2f} "
            f"(bid=${bid:.2f} ask=${ask:.2f}) | strike ${strike} exp {target_exp} δ={delta} | #{order_id}")
        await broadcast({"type": "trade_placed", "data": {
            "symbol": symbol, "option_symbol": opt_symbol,
            "side": "buy_to_open", "contracts": num_contracts,
            "strike": strike, "expiration": target_exp,
            "entry": limit_price, "order_id": order_id,
            "setup": alert["setup"], "direction": direction,
            "is_option": True,
        }})
        # Compute and store scale-out targets for this position
        scales = exit_manager.compute_scales(
            option_symbol = opt_symbol,
            underlying    = symbol,
            direction     = direction,
            cost_basis    = limit_price,   # what we paid per contract
            qty           = num_contracts,
            alert_target  = float(alert.get("target", 0)),
        )
        state["scale_targets"][opt_symbol] = scales
        add_log("ExitMgr",
            f"{opt_symbol}: scales set — "
            f"S1 {scales.scale1.qty}ct@+30-40% | "
            f"S2 {scales.scale2.qty}ct@+55-70% | "
            f"S3 {scales.scale3.qty}ct@+100-120%")
        return True
    return False


# ── Auto-trade ────────────────────────────────────

async def auto_trade(alert: dict):
    if not tradier or not AUTO_TRADE:
        return

    symbol = alert["symbol"]
    if symbol in state["traded_today"]:
        add_log("Execution", f"Skipping {symbol} — already traded today")
        return

    balances = await asyncio.to_thread(tradier.get_balances)
    if not balances:
        return

    total_equity  = balances.get("total_equity", 100000)
    option_bp     = balances.get("margin", {}).get("option_buying_power", total_equity)
    stock_bp      = balances.get("margin", {}).get("stock_buying_power", total_equity)

    if state["daily_pl"] < -(total_equity * MAX_DAILY_LOSS):
        add_log("Risk", f"Daily loss limit hit ({state['daily_pl']:+.2f}) — halting trades", "warning")
        return

    entry          = alert["entry"]
    stop           = alert["stop"]
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0:
        return
    risk_dollars = total_equity * MAX_RISK_PER_TRADE

    # Try options first
    if is_market_open():
        traded = await _find_option_and_trade(symbol, alert["direction"], entry, risk_dollars, alert)
        if traded:
            state["traded_today"].add(symbol)
            return

    # Fallback: equity order
    shares     = max(1, int(risk_dollars / risk_per_share))
    max_shares = max(1, int(stock_bp * 0.10 / entry))
    shares     = min(shares, max_shares)
    side       = "buy" if alert["direction"] == "call" else "sell_short"

    result = await asyncio.to_thread(
        tradier.place_equity_order,
        symbol, side, shares, "limit", entry, None, "gtc",
        f"tradeagent-{alert['setup']}",
    )
    if result and result.get("order", {}).get("status") == "ok":
        order_id = result["order"]["id"]
        state["traded_today"].add(symbol)
        add_log("Execution",
            f"Equity {side.upper()} {shares}x {symbol} @ ${entry:.2f} | #{order_id}")
        await broadcast({"type": "trade_placed", "data": {
            "symbol": symbol, "side": side, "shares": shares,
            "entry": entry, "stop": stop, "target": alert["target"],
            "order_id": order_id, "setup": alert["setup"],
            "direction": alert["direction"], "is_option": False,
        }})


# ── Scan loop ─────────────────────────────────────

async def run_scan():
    if state["scan_running"]:
        return
    state["scan_running"] = True
    add_log("Orchestrator",
        f"Scan started — {len(WATCHLIST)} symbols | {'OPEN' if is_market_open() else 'CLOSED'}")

    new_alerts = []

    # Fetch account context once per scan (shared across all symbols)
    _positions: list[dict] = []
    _account_equity: float = 0.0
    if tradier:
        try:
            _raw_pos = await asyncio.to_thread(tradier.get_positions)
            _enriched, _ = await _enrich_positions(_raw_pos)
            _positions = _enriched
            _balances = await asyncio.to_thread(tradier.get_balances)
            _account_equity = (_balances or {}).get("total_equity", 0.0)
        except Exception as _e:
            log.warning(f"Pre-scan account fetch failed: {_e}")

    _daily_pl_pct = (
        state["daily_pl"] / state["opening_equity"]
        if state["opening_equity"] else 0.0
    )

    scan_symbols = list(WATCHLIST)
    add_log("Orchestrator", f"Scanning {len(scan_symbols)} symbols (focused watchlist)")

    try:
        for symbol in scan_symbols:
            try:
                quote      = await asyncio.to_thread(get_quote, symbol)
                candles_5m = await asyncio.to_thread(get_candles, symbol, "5m", 3)
                candles_1h = await asyncio.to_thread(get_candles, symbol, "1h", 14)
                candles_1d = await asyncio.to_thread(get_candles, symbol, "1d", 60)
                news       = fh.get_news(symbol, days=1) if fh else []
                if quote:
                    state["quotes"][symbol] = quote

                # ── Multi-agent evaluation (3-layer pipeline) ───────
                alert = await orchestrator.evaluate(
                    symbol         = symbol,
                    candles_5m     = candles_5m,
                    candles_1h     = candles_1h,
                    candles_1d     = candles_1d,
                    quote          = quote,
                    news           = news,
                    positions      = _positions,
                    account_equity = _account_equity,
                    daily_pl_pct   = _daily_pl_pct,
                    traded_today   = state["traded_today"],
                )

                # Always store agent views (even for non-signal symbols) so
                # the dashboard agent-watchlist panel shows analysis for every
                # scanned ticker, not just ones that passed all gates.
                state["agent_views"][symbol] = orchestrator._views_cache.get(symbol, [])

                if alert:
                    ad = alert.to_dict()
                    if quote and quote.get("c"):
                        ad["current_price"] = round(float(quote["c"]), 2)
                    new_alerts.append(ad)

                    # Persist to 10-session signal history log
                    _append_to_history(ad)

                    # Record prediction for feedback loop
                    state["signal_log"][symbol] = {
                        "signal_id": alert.signal_id,
                        "alert_dict": ad,
                    }
                    feedback.record_prediction(
                        signal_id       = alert.signal_id,
                        symbol          = symbol,
                        direction       = alert.direction,
                        agent_views     = ad.get("agent_views", []),
                        consensus_score = alert.consensus_score,
                        entry_price     = alert.entry,
                    )

                    n_agents = sum(
                        1 for v in ad.get("agent_views", [])
                        if v.get("direction") == alert.direction
                    )
                    add_log("Orchestrator",
                        f"{symbol}: {alert.setup} ({alert.direction.upper()}) "
                        f"conf={alert.confidence:.0%} R:R=1:{alert.risk_reward} "
                        f"agents={n_agents}/7 consensus={alert.consensus_score:.0%}")
                    await broadcast({"type": "alert", "data": ad})
                    if is_market_open():
                        await auto_trade(ad)

            except Exception as e:
                log.error(f"Error scanning {symbol}: {e}")

        state["alerts"] = (new_alerts + state["alerts"])[:50]
        state["last_scan"] = datetime.now().isoformat()
        add_log("Orchestrator", f"Scan complete — {len(new_alerts)} new signal(s)")
        await snapshot_balance()
    finally:
        state["scan_running"] = False


async def scan_scheduler():
    """Scan every 15 minutes during market hours only (9:30–16:00 ET, Mon–Fri)."""
    while True:
        await asyncio.sleep(60)
        if is_market_open():
            await run_scan()


async def monitor_exits():
    """
    Background task: checks open options positions every 45s during market hours.

    For each option position:
      1. Auto-bootstraps scale targets if missing (handles restarts + manual trades).
      2. Stop-loss: closes entire position if option is down > 50% (setup invalidated).
      3. Scale-out: partial closes at 30-40% / 55-70% / 100-120% profit tiers.
    """
    while True:
        await asyncio.sleep(45)
        if not is_market_open() or not tradier:
            continue
        try:
            # ── Resolve any pending close orders first ─────────────────
            # Check Tradier order status for every tracked pending close.
            # If the order filled → confirm the scale. If cancelled/rejected
            # → clear it so we can retry. If still open → leave it alone.
            if state["pending_closes"]:
                all_orders = await asyncio.to_thread(tradier.get_orders)
                order_map  = {str(o.get("id")): o for o in (all_orders or [])}
                for _sym, pend in list(state["pending_closes"].items()):
                    oid  = str(pend.get("order_id", ""))
                    ordr = order_map.get(oid, {})
                    ostatus = ordr.get("status", "").lower()
                    if ostatus in ("filled", "partially_filled"):
                        filled_qty = int(ordr.get("filled_quantity") or ordr.get("quantity") or 0)
                        add_log("ExitMgr",
                            f"Order #{oid} FILLED: {filled_qty}x {_sym} "
                            f"({pend.get('type','exit')})")
                        state["pending_closes"].pop(_sym, None)
                    elif ostatus in ("canceled", "rejected", "expired"):
                        add_log("ExitMgr",
                            f"Order #{oid} {ostatus} for {_sym} — holding 10min before retry",
                            "warning")
                        # Replace with a timed hold — prevents immediate flood retry
                        state["pending_closes"][_sym] = {
                            "order_id":  None,
                            "qty":       pend.get("qty"),
                            "type":      pend.get("type", "exit") + "_retry_hold",
                            "placed_at": datetime.now().isoformat(),
                        }
                        # Un-fill scale so monitor retries after hold expires
                        sc = state["scale_targets"].get(_sym)
                        if sc:
                            for scale in [sc.scale1, sc.scale2, sc.scale3]:
                                if getattr(scale, "order_id", None) == oid:
                                    scale.filled = False
                    # Clear retry_hold entries older than 10 minutes
                    for _sym, pend in list(state["pending_closes"].items()):
                        if "_retry_hold" in pend.get("type", "") and pend.get("order_id") is None:
                            placed = pend.get("placed_at", "")
                            try:
                                age = (datetime.now() - datetime.fromisoformat(placed)).seconds
                                if age > 600:
                                    state["pending_closes"].pop(_sym, None)
                            except Exception:
                                state["pending_closes"].pop(_sym, None)

            raw_positions = await asyncio.to_thread(tradier.get_positions)
            positions, _ = await _enrich_positions(raw_positions)
            if not positions:
                continue

            for pos in positions:
                sym = pos.get("symbol", "")
                if not is_option_symbol(sym):
                    continue

                current_px = pos.get("current_price", 0)  # mid price from _enrich_positions
                qty_held   = int(pos.get("quantity", 0))
                if not current_px or qty_held <= 0:
                    continue

                # ── Auto-bootstrap missing scale targets ───────────────────
                # Handles: server restarts, manually placed trades, any position
                # opened before scale targets were stored.
                if sym not in state["scale_targets"]:
                    m = re.match(r'^([A-Z]{1,6})\d{6}([CP])\d{8}$', sym)
                    if not m:
                        continue
                    underlying_sym = m.group(1)
                    direction_sym  = "call" if m.group(2) == "C" else "put"
                    total_cost     = pos.get("cost_basis", 0)
                    if not total_cost:
                        continue
                    per_share_cost = round(total_cost / (qty_held * 100), 4)
                    state["scale_targets"][sym] = exit_manager.compute_scales(
                        option_symbol = sym,
                        underlying    = underlying_sym,
                        direction     = direction_sym,
                        cost_basis    = per_share_cost,
                        qty           = qty_held,
                        alert_target  = 0.0,  # unknown for manual trades
                    )
                    add_log("ExitMgr",
                        f"Auto-registered {sym}: cost=${per_share_cost:.2f}/sh "
                        f"qty={qty_held} dir={direction_sym}")

                scales: ScaleTargets = state["scale_targets"].get(sym)
                if not scales or scales.cost_basis <= 0:
                    continue

                pnl_pct = (current_px - scales.cost_basis) / scales.cost_basis

                # ── Skip if a close order is already pending ───────────────
                if sym in state["pending_closes"]:
                    continue

                # ── Stop-loss: close full position if down > 50% ──────────
                # Use market order — guaranteed fill regardless of spread.
                all_filled = scales.scale1.filled and scales.scale2.filled and scales.scale3.filled
                if pnl_pct < -0.50 and not all_filled:
                    result  = await asyncio.to_thread(
                        tradier.place_option_order,
                        sym, "sell_to_close", qty_held, "market", None, "day",
                    )
                    if result and result.get("order", {}).get("status") == "ok":
                        order_id = result.get("order", {}).get("id")
                        state["pending_closes"][sym] = {
                            "order_id":  order_id,
                            "qty":       qty_held,
                            "type":      "stop_loss",
                            "placed_at": datetime.now().isoformat(),
                        }
                        add_log("ExitMgr",
                            f"STOP LOSS market order placed: {qty_held}x {sym} "
                            f"({pnl_pct:+.0%}) — premium >50% loss — order #{order_id}",
                            "warning")
                        await broadcast({"type": "stop_loss", "data": {
                            "symbol": scales.underlying, "option_symbol": sym,
                            "qty": qty_held, "pnl_pct": round(pnl_pct, 3),
                        }})
                    else:
                        # Order rejected — don't retry for 10 min to avoid flooding
                        state["pending_closes"][sym] = {
                            "order_id":  None,
                            "qty":       qty_held,
                            "type":      "stop_loss_retry_hold",
                            "placed_at": datetime.now().isoformat(),
                        }
                        add_log("ExitMgr",
                            f"STOP LOSS order rejected for {sym} — holding 10min before retry",
                            "warning")
                    continue

                # ── Scale-out (profit taking) ──────────────────────────────
                underlying    = scales.underlying
                quote         = state["quotes"].get(underlying, {})
                candles_5m    = await asyncio.to_thread(get_candles, underlying, "5m", 1)
                underlying_px = float(quote.get("c", current_px))

                for scale_name, scale in [
                    ("scale1", scales.scale1),
                    ("scale2", scales.scale2),
                    ("scale3", scales.scale3),
                ]:
                    if scale.filled or scale.qty <= 0:
                        continue
                    if qty_held < scale.qty:
                        continue

                    trigger, reason = exit_manager.should_trigger_scale(
                        scale, pnl_pct, candles_5m, quote,
                        scales.direction, underlying_px, scales.alert_target,
                    )
                    if not trigger:
                        continue

                    result  = await asyncio.to_thread(
                        tradier.place_option_order,
                        sym, "sell_to_close", scale.qty, "market", None, "day",
                    )
                    if result and result.get("order", {}).get("status") == "ok":
                        from datetime import datetime as _dt
                        order_id = result.get("order", {}).get("id")
                        scale.order_id  = order_id
                        scale.filled_at = _dt.now().isoformat()
                        state["pending_closes"][sym] = {
                            "order_id":  order_id,
                            "qty":       scale.qty,
                            "type":      scale_name,
                            "placed_at": _dt.now().isoformat(),
                        }
                        add_log("ExitMgr",
                            f"{scale_name.upper()} market close placed: {scale.qty}x {sym} "
                            f"({pnl_pct:+.0%}) — {reason} — order #{order_id}")
                        await broadcast({"type": "scale_exit", "data": {
                            "symbol": underlying, "option_symbol": sym,
                            "scale": scale_name, "qty": scale.qty,
                            "price": sell_px, "pnl_pct": round(pnl_pct, 3),
                            "reason": reason,
                        }})
                        # Mark filled only after we confirm via order status next cycle
                        # but prevent re-triggering in the meantime
                        scale.filled = True

                        if scale_name == "scale3" and not scales.roll_evaluated:
                            scales.roll_evaluated = True
                            proceeds = sell_px * scale.qty * 100
                            balances = await asyncio.to_thread(tradier.get_balances)
                            equity   = (balances or {}).get("total_equity", 10000)
                            views    = state["agent_views"].get(underlying, [])
                            roll     = exit_manager.evaluate_roll(
                                underlying, views, equity, proceeds)
                            if roll.should_roll:
                                add_log("ExitMgr",
                                    f"ROLL signal: {underlying} — {roll.rationale}")
                                await broadcast({"type": "roll_signal", "data": {
                                    "symbol": underlying,
                                    "roll_dollars": roll.roll_dollars,
                                    "conviction": roll.conviction,
                                    "rationale": roll.rationale,
                                }})
                            else:
                                add_log("ExitMgr",
                                    f"No roll for {underlying} — {roll.rationale}")
                        break  # one scale order per position per cycle

        except Exception as e:
            log.error(f"monitor_exits error: {e}")


# ── Daily CSV export ──────────────────────────────

# Columns written to every daily CSV (in order)
_CSV_COLUMNS = [
    "session_date", "timestamp", "signal_id",
    "symbol", "setup", "direction",
    "entry", "stop", "target",
    "confidence_pct", "consensus_pct", "risk_reward",
    "timeframe",
    "agent_momentum", "agent_quant", "agent_options_flow",
    "agent_sentiment", "agent_fundamental", "agent_macro", "agent_stat_arb",
    "signals_text",
]

def _build_csv_row(sig: dict) -> dict:
    """Flatten one signal dict into a CSV-ready row."""
    # Pull per-agent directions from agent_views list
    agent_dirs = {v.get("agent", ""): v.get("direction", "neutral")
                  for v in sig.get("agent_views", [])}
    signals_text = " | ".join(
        s for s in sig.get("signals", []) if isinstance(s, str)
    )
    return {
        "session_date":       sig.get("session_date", ""),
        "timestamp":          sig.get("timestamp", ""),
        "signal_id":          sig.get("signal_id", ""),
        "symbol":             sig.get("symbol", ""),
        "setup":              sig.get("setup", ""),
        "direction":          sig.get("direction", ""),
        "entry":              sig.get("entry", ""),
        "stop":               sig.get("stop", ""),
        "target":             sig.get("target", ""),
        "confidence_pct":     sig.get("confidence_pct", round((sig.get("confidence", 0)) * 100)),
        "consensus_pct":      sig.get("consensus_pct", round((sig.get("consensus_score", 0)) * 100)),
        "risk_reward":        sig.get("risk_reward", ""),
        "timeframe":          sig.get("timeframe", ""),
        "agent_momentum":     agent_dirs.get("momentum", ""),
        "agent_quant":        agent_dirs.get("quant", ""),
        "agent_options_flow": agent_dirs.get("options_flow", ""),
        "agent_sentiment":    agent_dirs.get("sentiment", ""),
        "agent_fundamental":  agent_dirs.get("fundamental", ""),
        "agent_macro":        agent_dirs.get("macro", ""),
        "agent_stat_arb":     agent_dirs.get("stat_arb", ""),
        "signals_text":       signals_text,
    }

def export_daily_csv(target_date: str | None = None) -> Path:
    """
    Write a CSV for target_date (ISO string, defaults to today) using
    signals stored in signal_history.json.  Returns the Path written.
    """
    if target_date is None:
        target_date = date.today().isoformat()

    history   = _load_signal_history()
    day_sigs  = [s for s in history if s.get("session_date") == target_date]

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = _LOGS_DIR / f"signals_{target_date}.csv"

    with csv_path.open("w", newline="") as fh_:
        writer = csv.DictWriter(fh_, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for sig in day_sigs:
            writer.writerow(_build_csv_row(sig))

    log.info(f"Daily CSV written: {csv_path} ({len(day_sigs)} signals)")
    add_log("System", f"Daily CSV exported: signals_{target_date}.csv ({len(day_sigs)} signals)")
    return csv_path


async def daily_csv_scheduler():
    """
    Fires once per weekday at 16:05 ET (5 min after market close).
    Exports the day's signals to logs/signals_YYYY-MM-DD.csv.
    Also catches up if the server was offline during the close window.
    """
    _last_export: str | None = None
    while True:
        await asyncio.sleep(60)
        now_et = datetime.now(ET)
        today  = now_et.date().isoformat()
        # Weekday only (Mon=0 … Fri=4), between 16:05 and 16:30 ET
        if (now_et.weekday() < 5
                and now_et.hour == 16
                and 5 <= now_et.minute <= 30
                and _last_export != today):
            try:
                export_daily_csv(today)
                _last_export = today
            except Exception as e:
                log.error(f"daily_csv_scheduler error: {e}")


# ── Lifecycle ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    add_log("System", "TradeAgent starting up")
    add_log("System",
        f"Data: yfinance | News: {'Finnhub' if fh else 'off'} | "
        f"Trading: {'paper' if _tradier_paper else 'LIVE'} | auto={AUTO_TRADE}")
    await snapshot_balance()
    asyncio.create_task(run_scan())
    asyncio.create_task(scan_scheduler())
    asyncio.create_task(monitor_exits())
    asyncio.create_task(daily_csv_scheduler())
    yield
    add_log("System", "TradeAgent shutting down")


app = FastAPI(title="TradeAgent API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"


# ── Static ────────────────────────────────────────

@app.get("/")
async def serve_dashboard():
    index = DASHBOARD_DIR / "index.html"
    return FileResponse(index) if index.exists() else JSONResponse({"status": "running"})

@app.get("/options")
async def serve_options():
    page = DASHBOARD_DIR / "options.html"
    return FileResponse(page) if page.exists() else JSONResponse({"error": "options.html not found"})


# ── Market data ───────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "agent_status":      "running",
        "market_open":       is_market_open(),
        "scan_running":      state["scan_running"],
        "last_scan":         state["last_scan"],
        "active_alerts":     len(state["alerts"]),
        "quotes_cached":     len(state["quotes"]),
        "connected_clients": len(state["connected_ws"]),
        "finnhub_news":      fh is not None,
        "tradier_connected": tradier is not None,
        "tradier_paper":     _tradier_paper,
        "auto_trade":        AUTO_TRADE,
        "daily_pl":          state["daily_pl"],
    }

@app.get("/api/signals")
async def get_signals(limit: int = 50):
    return {"signals": state["alerts"][:limit], "last_scan": state["last_scan"]}

@app.get("/api/signals/history")
async def get_signal_history():
    """
    Returns all signals from the last 10 trading sessions, grouped by date.
    Most recent session first.
    """
    history = _load_signal_history()
    # Group by session_date
    from collections import defaultdict
    by_date: dict[str, list] = defaultdict(list)
    for sig in history:
        by_date[sig.get("session_date", "unknown")].append(sig)
    sessions = [
        {"date": d, "signals": by_date[d]}
        for d in sorted(by_date.keys(), reverse=True)
    ]
    return {"sessions": sessions, "total": len(history)}

@app.get("/api/signals/export")
async def export_signals_csv(target_date: str | None = None):
    """
    Trigger an immediate CSV export for the given date (YYYY-MM-DD, defaults
    to today).  Returns the file as a download attachment.
    """
    d = target_date or date.today().isoformat()
    try:
        csv_path = await asyncio.to_thread(export_daily_csv, d)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    if not csv_path.exists():
        return JSONResponse({"error": "No signals found for that date"}, status_code=404)
    from fastapi.responses import FileResponse as _FR
    return _FR(
        path=str(csv_path),
        media_type="text/csv",
        filename=csv_path.name,
        headers={"Content-Disposition": f'attachment; filename="{csv_path.name}"'},
    )

@app.get("/api/quote/{symbol}")
async def get_quote_endpoint(symbol: str):
    symbol = symbol.upper()
    cached = state["quotes"].get(symbol)
    if cached:
        return cached
    q = await asyncio.to_thread(get_quote, symbol)
    if q:
        state["quotes"][symbol] = q
        return q
    return JSONResponse({"error": "Symbol not found"}, status_code=404)

@app.get("/api/quotes")
async def get_all_quotes():
    return {"quotes": state["quotes"]}

@app.get("/api/scan")
async def trigger_scan(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scan)
    return {"status": "scan triggered", "symbols": len(WATCHLIST)}

@app.get("/api/scan/status")
async def scan_status():
    return {
        "scanning":        state["scan_running"],
        "symbols_analyzed": len(state["agent_views"]),
        "last_scan":       state.get("last_scan"),
    }

@app.get("/api/equity")
async def get_equity():
    return {"history": state["equity_history"], "daily_pl": state["daily_pl"],
            "opening_equity": state["opening_equity"]}

@app.get("/api/watchlist")
async def get_watchlist():
    return {"symbols": WATCHLIST}

@app.get("/api/agents/status")
async def get_agent_status():
    """Current agent weights, recent accuracy, pending count, and regime."""
    regime_info = None
    last_regime = orchestrator.get_last_regime()
    if last_regime:
        regime_info = {
            "label":             last_regime.label,
            "vix":               last_regime.vix,
            "adx":               last_regime.adx,
            "realized_vol":      last_regime.realized_vol,
            "description":       last_regime.description,
            "weight_multipliers": last_regime.weight_multipliers,
        }
    return {
        "weights":         feedback.get_weights(),
        "base_weights":    dict(_arb_module.BASE_WEIGHTS),
        "recent_accuracy": feedback.get_recent_accuracy(n=50),
        "pending_trades":  len(feedback.get_pending()),
        "signal_log":      {sym: v["signal_id"] for sym, v in state["signal_log"].items()},
        "regime":          regime_info,
    }

@app.patch("/api/agents/weights")
async def update_agent_weights(payload: dict):
    """Update base weights for one or more agents in real time."""
    VALID = {"momentum", "sentiment", "fundamental", "quant",
             "macro", "options_flow", "stat_arb"}
    MIN_W, MAX_W = 0.1, 10.0
    updated = {}
    for agent, val in payload.items():
        if agent not in VALID:
            continue
        clamped = round(max(MIN_W, min(MAX_W, float(val))), 2)
        _arb_module.BASE_WEIGHTS[agent] = clamped   # authoritative dict
        updated[agent] = clamped
    if updated:
        add_log("System", f"Agent base weights updated: {updated}")
    return {"base_weights": dict(_arb_module.BASE_WEIGHTS), "updated": updated}

@app.get("/api/agents/presets")
async def get_presets():
    """List all available regime presets and their weight configurations."""
    from agents.arbitration import REGIME_PRESETS
    return {
        "presets": REGIME_PRESETS,
        "current_weights": dict(_arb_module.BASE_WEIGHTS),
    }

@app.post("/api/agents/preset/{name}")
async def apply_preset(name: str):
    """Apply a named weight preset, updating BASE_WEIGHTS immediately."""
    from agents.arbitration import apply_preset as _apply, REGIME_PRESETS
    if name not in REGIME_PRESETS:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Unknown preset '{name}'")
    updated = _apply(name)
    add_log("System", f"Preset applied: {name} → {updated}")
    await broadcast({"type": "preset_applied", "data": {"preset": name, "weights": updated}})
    return {"preset": name, "base_weights": updated}

@app.get("/api/scale_targets")
async def get_scale_targets():
    """Current scale-out targets for all tracked option positions."""
    return {
        sym: exit_manager.to_dict(st)
        for sym, st in state["scale_targets"].items()
    }

@app.get("/api/agents/history")
async def get_agent_history(limit: int = 50):
    """Completed trades with per-agent reasoning and outcomes."""
    return {
        "history":      feedback.get_history(n=limit),
        "all_trades":   feedback.get_all_trades(n=limit),
        "total_trades": len(feedback.get_all_trades(n=500)),
    }

@app.get("/api/agents/views")
async def get_agent_views():
    """Latest agent views per symbol (from last scan)."""
    return {"agent_views": state["agent_views"]}

@app.get("/api/agents/views/{symbol}")
async def get_agent_views_symbol(symbol: str):
    """Latest agent views for a specific symbol."""
    sym = symbol.upper()
    views = state["agent_views"].get(sym, [])
    return {"symbol": sym, "agent_views": views}

@app.get("/api/sp500")
async def get_sp500():
    symbols = await asyncio.to_thread(get_sp500_symbols)
    return {"symbols": symbols, "count": len(symbols)}

@app.post("/api/scan")
async def trigger_scan_post(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scan)
    return {"status": "scan triggered", "symbols": len(WATCHLIST)}

@app.get("/api/news/{symbol}")
async def get_news(symbol: str, days: int = 1):
    symbol = symbol.upper()
    if fh:
        return {"symbol": symbol, "articles": fh.get_news(symbol, days=days)[:20]}
    return {"symbol": symbol, "articles": []}

@app.get("/api/logs")
async def get_logs(limit: int = 100):
    return {"logs": list(state["log_entries"])[:limit]}


# ── Tradier helpers ───────────────────────────────

def _no_tradier():
    if not tradier:
        return JSONResponse(
            {"error": "Tradier not configured"},
            status_code=503,
        )
    return None


async def _enrich_positions(positions: list[dict]) -> tuple[list[dict], float]:
    enriched, net_pl = [], 0.0
    for pos in positions:
        sym   = pos.get("symbol", "")
        qty   = pos.get("quantity", 0)
        cost  = pos.get("cost_basis", 0)

        if is_option_symbol(sym):
            # Get live option quote — use mid price (bid+ask)/2 for accurate
            # mark-to-market; "last" is often stale in sandbox/low-volume options.
            q = await asyncio.to_thread(tradier.get_option_quote, sym)
            if q:
                bid  = q.get("bid")  or 0.0
                ask  = q.get("ask")  or 0.0
                last = q.get("last") or 0.0
                if bid and ask:
                    current_price = round((bid + ask) / 2, 4)  # mid price — best current estimate
                else:
                    current_price = round(bid or ask or last, 4)  # fallback to any available price
            else:
                current_price = 0.0
            multiplier = 100
        else:
            current_price = state["quotes"].get(sym, {}).get("c")
            if not current_price:
                q = await asyncio.to_thread(get_quote, sym)
                if q:
                    state["quotes"][sym] = q
                    current_price = q.get("c")
            multiplier = 1

        if current_price and qty and cost:
            mktval = round(current_price * abs(qty) * multiplier, 2)
            unpl   = round(mktval - cost if qty > 0 else cost - mktval, 2)
            unpl_pct = round((unpl / cost) * 100, 2) if cost else 0
            pos["current_price"]     = current_price
            pos["market_value"]      = mktval
            pos["unrealized_pl"]     = unpl
            pos["unrealized_pl_pct"] = unpl_pct
            pos["is_option"]         = is_option_symbol(sym)
            net_pl += unpl

        # Annotate with pending close order info so UI can show "closing" status
        pend = state["pending_closes"].get(sym)
        if pend:
            pos["pending_close"]     = True
            pos["pending_close_qty"] = pend.get("qty", 0)
            pos["pending_close_type"] = pend.get("type", "exit")
            pos["pending_order_id"]  = pend.get("order_id")

        enriched.append(pos)
    return enriched, round(net_pl, 2)


# ── Account endpoints ─────────────────────────────

@app.get("/api/balances")
async def get_balances():
    e = _no_tradier()
    if e: return e
    b = await asyncio.to_thread(tradier.get_balances)
    return b or JSONResponse({"error": "Failed"}, status_code=502)

@app.get("/api/positions")
async def get_positions():
    e = _no_tradier()
    if e: return e
    positions = await asyncio.to_thread(tradier.get_positions)
    enriched, net_pl = await _enrich_positions(positions)
    return {"positions": enriched, "net_unrealized_pl": net_pl}

@app.get("/api/orders")
async def get_orders():
    e = _no_tradier()
    if e: return e
    orders = await asyncio.to_thread(tradier.get_orders)
    return {"orders": orders}


# ── Close position ────────────────────────────────

@app.post("/api/positions/{symbol}/close")
async def close_position(symbol: str):
    e = _no_tradier()
    if e: return e

    positions = await asyncio.to_thread(tradier.get_positions)
    # URL-decode + uppercase — options symbols may have special chars
    sym = symbol.upper().replace("%20", " ")
    pos = next((p for p in positions if p.get("symbol", "").upper() == sym), None)
    if not pos:
        return JSONResponse({"error": f"No open position for {sym}"}, status_code=404)

    qty    = pos.get("quantity", 0)
    result = await asyncio.to_thread(tradier.close_position, sym, qty)
    if not result:
        return JSONResponse({"error": "Close order failed"}, status_code=502)

    pos_type = "option" if is_option_symbol(sym) else "equity"
    add_log("Execution", f"Closed {pos_type} position: {sym} ({int(qty):+d} shares/contracts)")
    await broadcast({"type": "position_closed", "symbol": sym})

    # ── Feedback: record outcome ──────────────────────────────────
    log_entry = state["signal_log"].get(sym)
    if log_entry:
        try:
            enriched, _ = await _enrich_positions([pos])
            ep = enriched[0] if enriched else {}
            pnl_dollars = ep.get("unrealized_pl", 0.0) or 0.0
            cost_basis  = pos.get("cost_basis") or 1.0
            pnl_pct     = round(pnl_dollars / cost_basis * 100, 2) if cost_basis else 0.0
            exit_price  = ep.get("current_price") or 0.0
            feedback.record_outcome(
                signal_id   = log_entry["signal_id"],
                exit_price  = exit_price,
                pnl_dollars = pnl_dollars,
                pnl_pct     = pnl_pct,
            )
            add_log("System",
                f"Feedback recorded: {sym} P&L={pnl_dollars:+.2f} ({pnl_pct:+.1f}%) "
                f"→ {'WIN' if pnl_dollars > 0 else 'LOSS'}")
            del state["signal_log"][sym]
        except Exception as e:
            log.warning(f"Feedback recording failed for {sym}: {e}")

    return result


# ── Equity order endpoints ────────────────────────

class OrderRequest(BaseModel):
    symbol:     str
    side:       str
    quantity:   int
    order_type: str   = "market"
    price:      float = None
    stop:       float = None
    duration:   str   = "day"
    tag:        str   = "tradeagent"

@app.post("/api/orders")
async def place_order(req: OrderRequest):
    e = _no_tradier()
    if e: return e
    result = await asyncio.to_thread(
        tradier.place_equity_order,
        req.symbol, req.side, req.quantity,
        req.order_type, req.price, req.stop, req.duration, req.tag,
    )
    if not result:
        return JSONResponse({"error": "Order failed"}, status_code=502)
    add_log("Execution", f"Manual order: {req.side} {req.quantity}x {req.symbol}")
    return result

@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: int):
    e = _no_tradier()
    if e: return e
    result = await asyncio.to_thread(tradier.cancel_order, order_id)
    return result or JSONResponse({"error": "Cancel failed"}, status_code=502)


# ── Options endpoints ─────────────────────────────

@app.get("/api/options/expirations/{symbol}")
async def option_expirations(symbol: str):
    e = _no_tradier()
    if e: return e
    exps = await asyncio.to_thread(tradier.get_option_expirations, symbol.upper())
    return {"symbol": symbol.upper(), "expirations": exps}

@app.get("/api/options/chain/{symbol}")
async def option_chain(symbol: str, expiration: str = None):
    e = _no_tradier()
    if e: return e

    sym  = symbol.upper()
    exps = await asyncio.to_thread(tradier.get_option_expirations, sym)
    if not exps:
        return {"symbol": sym, "expiration": None, "calls": [], "puts": []}

    # Default to nearest expiration ≥ 5 days out
    if not expiration:
        today_date = date.today()
        for exp_str in sorted(exps):
            try:
                if (date.fromisoformat(exp_str) - today_date).days >= 5:
                    expiration = exp_str
                    break
            except ValueError:
                pass
    if not expiration:
        expiration = exps[0]

    chain = await asyncio.to_thread(tradier.get_option_chain, sym, expiration)

    # Get current price for this symbol
    spot = state["quotes"].get(sym, {}).get("c")
    if not spot:
        q = await asyncio.to_thread(get_quote, sym)
        spot = q.get("c") if q else 0

    calls = sorted([o for o in chain if o.get("option_type") == "call"],
                   key=lambda o: o.get("strike", 0))
    puts  = sorted([o for o in chain if o.get("option_type") == "put"],
                   key=lambda o: o.get("strike", 0))

    return {
        "symbol":      sym,
        "expiration":  expiration,
        "expirations": exps,
        "spot":        spot,
        "calls":       calls,
        "puts":        puts,
    }


@app.get("/api/options/preview/{symbol}")
async def preview_option_contract(
    symbol:       str,
    direction:    str   = "call",
    entry:        float = 0,
    risk_dollars: float = 2000,
):
    """
    Resolve the best matching options contracts (weekly + swing) for a signal
    without placing an order. Used by the buy confirmation modal.
    """
    e = _no_tradier()
    if e: return e
    sym = symbol.upper()

    if entry <= 0:
        q = await asyncio.to_thread(get_quote, sym)
        entry = float((q or {}).get("c", 0))

    expirations = await asyncio.to_thread(tradier.get_option_expirations, sym)
    if not expirations:
        return JSONResponse({"error": "No expirations found"}, status_code=404)

    today_date = date.today()
    weekly_exp, swing_exp = None, None
    for exp_str in sorted(expirations):
        try:
            exp_date = date.fromisoformat(exp_str)
        except ValueError:
            continue
        days_out = (exp_date - today_date).days
        if 5 <= days_out <= 14 and weekly_exp is None:
            weekly_exp = exp_str
        if 14 < days_out <= 35 and swing_exp is None:
            swing_exp = exp_str

    results = []
    for exp_str, trade_type in [(weekly_exp, "weekly"), (swing_exp, "swing")]:
        if not exp_str:
            continue
        chain = await asyncio.to_thread(tradier.get_option_chain, sym, exp_str)
        opt_type  = "call" if direction == "call" else "put"
        contracts = [o for o in chain if o.get("option_type") == opt_type and (o.get("ask") or 0) > 0]
        if not contracts:
            continue
        contracts.sort(key=lambda o: abs((o.get("strike") or 0) - entry))
        contract = contracts[0]
        ask = float(contract.get("ask", 0))
        strike = contract.get("strike", 0)
        opt_symbol = contract.get("symbol", "")
        greeks = contract.get("greeks") or {}
        delta = round(float(greeks.get("delta", 0) or 0), 2)
        iv    = round(float(greeks.get("iv", 0) or 0) * 100, 1)
        cost_per_contract = ask * 100
        num_contracts = max(1, min(10, int(risk_dollars / cost_per_contract) if cost_per_contract > 0 else 1))
        limit_price = round(ask * 1.02, 2)
        total_cost  = round(limit_price * 100 * num_contracts, 2)
        results.append({
            "trade_type":    trade_type,
            "option_symbol": opt_symbol,
            "underlying":    sym,
            "direction":     direction,
            "strike":        strike,
            "expiration":    exp_str,
            "ask":           ask,
            "limit_price":   limit_price,
            "num_contracts": num_contracts,
            "total_cost":    total_cost,
            "delta":         delta,
            "iv":            iv,
            "days_to_expiry": (date.fromisoformat(exp_str) - today_date).days,
        })

    if not results:
        return JSONResponse({"error": "No contracts found for this symbol/direction"}, status_code=404)

    return {"symbol": sym, "direction": direction, "entry": entry, "contracts": results}


class OptionOrderRequest(BaseModel):
    option_symbol: str
    side:          str           # buy_to_open | sell_to_close | buy_to_close | sell_to_open
    quantity:      int
    order_type:    str   = "limit"
    price:         float = None
    duration:      str   = "day"


# ── Backtest ──────────────────────────────────────

class BacktestRequest(BaseModel):
    symbols:       list[str] = []   # empty = WATCHLIST
    lookback_days: int        = 60
    preset:        str | None = None

@app.post("/api/backtest/run")
async def run_backtest_endpoint(req: BacktestRequest):
    """
    Walk-forward daily backtest for the multi-agent pipeline.
    Runs in a thread to avoid blocking the event loop.
    """
    from agents.backtester import run_backtest
    symbols = [s.upper() for s in req.symbols] if req.symbols else list(WATCHLIST)
    symbols = list(dict.fromkeys(symbols))[:50]   # dedup, cap at 50
    add_log("Backtest", f"Starting backtest: {len(symbols)} symbols | {req.lookback_days}d | preset={req.preset}")
    result = await asyncio.to_thread(
        run_backtest,
        symbols,
        max(10, min(90, req.lookback_days)),
        req.preset,
    )
    add_log("Backtest",
        f"Done: {result.signals_found} signals | WR={result.win_rate:.0%} | "
        f"avgRet={result.avg_return*100:+.2f}% | {result.run_time_sec}s")
    return result.to_dict()

@app.post("/api/options/orders")
async def place_option_order_endpoint(req: OptionOrderRequest):
    e = _no_tradier()
    if e: return e
    result = await asyncio.to_thread(
        tradier.place_option_order,
        req.option_symbol, req.side, req.quantity,
        req.order_type, req.price, req.duration,
    )
    if not result:
        return JSONResponse({"error": "Option order failed"}, status_code=502)
    add_log("Execution", f"Option order: {req.side} {req.quantity}x {req.option_symbol}")
    return result


# ── WebSocket ─────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state["connected_ws"].add(ws)
    add_log("WebSocket", f"Client connected ({len(state['connected_ws'])} total)")

    balances = await asyncio.to_thread(tradier.get_balances) if tradier else {}
    try:
        await ws.send_json({
            "type":     "init",
            "signals":  state["alerts"][:20],
            "quotes":   state["quotes"],
            "logs":     list(state["log_entries"])[:50],
            "equity":   state["equity_history"],
            "balances": balances or {},
            "daily_pl": state["daily_pl"],
            "status": {
                "market_open":   is_market_open(),
                "last_scan":     state["last_scan"],
                "auto_trade":    AUTO_TRADE,
                "tradier_paper": _tradier_paper,
            },
        })
        while True:
            balances = await asyncio.to_thread(tradier.get_balances) if tradier else {}
            await ws.send_json({
                "type":        "tick",
                "quotes":      state["quotes"],
                "balances":    balances or {},
                "daily_pl":    state["daily_pl"],
                "market_open": is_market_open(),
                "scan_running": state["scan_running"],
                "last_scan":   state["last_scan"],
            })
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        state["connected_ws"].discard(ws)
        add_log("WebSocket", f"Client disconnected ({len(state['connected_ws'])} total)")
