"""
agents/feedback.py
──────────────────
FeedbackEngine — trade outcome tracking + adaptive agent weight recomputation.

Storage: data/trade_memory.json (atomic writes, no SQLite dependency)
Window:  last 50 completed trades used for weight recomputation.

Weight range: [0.5, 1.5] mapping accuracy [0%, 100%] linearly.
  • 50% accuracy → weight 1.0 (neutral)
  • 100% accuracy → weight 1.5 (rewarded)
  • 0% accuracy   → weight 0.5 (penalized)
"""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

MEMORY_FILE   = Path(__file__).parent.parent / "data" / "trade_memory.json"
AGENT_NAMES   = ("momentum", "sentiment", "fundamental", "quant",
                  "macro", "options_flow", "stat_arb")
WINDOW        = 50
DEFAULT_WEIGHT = 1.0
MIN_TRADES_FOR_WEIGHT = 5   # need at least this many to deviate from default


def generate_signal_id(symbol: str) -> str:
    ts     = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{symbol}-{ts}-{suffix}"


class FeedbackEngine:
    def __init__(self):
        self._data = self._load()
        self._weights: dict[str, float] = dict(self._data.get("agent_weights", {}))
        if not self._weights:
            self._weights = {a: DEFAULT_WEIGHT for a in AGENT_NAMES}

    # ── Persistence ───────────────────────────────────────────────

    def _load(self) -> dict:
        if MEMORY_FILE.exists():
            try:
                return json.loads(MEMORY_FILE.read_text())
            except Exception as e:
                log.warning(f"FeedbackEngine: failed to load memory: {e}")
        return {"agent_weights": {}, "trades": []}

    def _save(self):
        self._data["agent_weights"] = self._weights
        tmp = MEMORY_FILE.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._data, indent=2))
            os.replace(tmp, MEMORY_FILE)
        except Exception as e:
            log.error(f"FeedbackEngine: save failed: {e}")

    # ── Public API ────────────────────────────────────────────────

    def record_prediction(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        agent_views: list[dict],
        consensus_score: float,
        entry_price: float,
    ) -> None:
        """Called when a signal fires and enters the signal_log."""
        record = {
            "signal_id":      signal_id,
            "symbol":         symbol,
            "direction":      direction,
            "consensus_score": round(consensus_score, 3),
            "entry_price":    entry_price,
            "agent_views":    agent_views,
            "predicted_at":   datetime.now().isoformat(),
            "exit_price":     None,
            "pnl_dollars":    None,
            "pnl_pct":        None,
            "outcome":        None,
            "closed_at":      None,
        }
        self._data["trades"].append(record)
        # Keep last 200 records to prevent unbounded growth
        if len(self._data["trades"]) > 200:
            self._data["trades"] = self._data["trades"][-200:]
        self._save()
        log.info(f"FeedbackEngine: prediction recorded {signal_id}")

    def record_outcome(
        self,
        signal_id: str,
        exit_price: float,
        pnl_dollars: float,
        pnl_pct: float,
    ) -> None:
        """Called when a position is closed via the dashboard."""
        for trade in self._data["trades"]:
            if trade["signal_id"] == signal_id:
                trade["exit_price"]  = exit_price
                trade["pnl_dollars"] = round(pnl_dollars, 2)
                trade["pnl_pct"]     = round(pnl_pct, 2)
                trade["outcome"]     = "win" if pnl_dollars > 0 else "loss"
                trade["closed_at"]   = datetime.now().isoformat()
                log.info(f"FeedbackEngine: outcome recorded {signal_id} → {trade['outcome']} P&L={pnl_dollars:+.2f}")
                break
        self._recompute_weights()
        self._save()

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def get_recent_accuracy(self, n: int = 50) -> dict:
        """Per-agent accuracy stats for the /api/agents/status endpoint."""
        completed = [t for t in self._data["trades"] if t.get("outcome")][-n:]
        if not completed:
            return {a: {"accuracy": None, "total": 0, "correct": 0,
                        "avg_pnl": 0, "weight": self._weights.get(a, DEFAULT_WEIGHT)}
                    for a in AGENT_NAMES}

        stats: dict[str, dict] = {a: {"correct": 0, "total": 0, "pnl_sum": 0.0}
                                   for a in AGENT_NAMES}
        for trade in completed:
            trade_correct = trade["outcome"] == "win"
            pnl = trade.get("pnl_pct") or 0.0
            for view in trade.get("agent_views", []):
                agent = view.get("agent")
                if agent not in stats:
                    continue
                stats[agent]["total"] += 1
                # Agent correct if: (agreed AND win) OR (disagreed AND loss)
                agreed = (view.get("direction") == trade["direction"]
                          or view.get("direction") == "neutral")
                if (agreed and trade_correct) or (not agreed and not trade_correct):
                    stats[agent]["correct"] += 1
                stats[agent]["pnl_sum"] += pnl

        return {
            agent: {
                "accuracy": round(s["correct"] / s["total"] * 100, 1) if s["total"] else None,
                "total":    s["total"],
                "correct":  s["correct"],
                "avg_pnl":  round(s["pnl_sum"] / s["total"], 2) if s["total"] else 0,
                "weight":   round(self._weights.get(agent, DEFAULT_WEIGHT), 3),
            }
            for agent, s in stats.items()
        }

    def get_history(self, n: int = 50) -> list[dict]:
        """Last n completed trades (most recent first) for /api/agents/history."""
        completed = [t for t in self._data["trades"] if t.get("outcome")]
        return list(reversed(completed[-n:]))

    def get_pending(self) -> list[dict]:
        """Predictions that haven't been closed yet."""
        return [t for t in self._data["trades"] if not t.get("outcome")]

    def get_all_trades(self, n: int = 100) -> list[dict]:
        """All trades (pending + completed), most recent first."""
        return list(reversed(self._data["trades"][-n:]))

    # ── Weight recomputation ──────────────────────────────────────

    def _recompute_weights(self):
        completed = [t for t in self._data["trades"] if t.get("outcome")][-WINDOW:]
        if not completed:
            return

        for agent in AGENT_NAMES:
            correct = 0
            total   = 0
            for trade in completed:
                trade_correct = trade["outcome"] == "win"
                for view in trade.get("agent_views", []):
                    if view.get("agent") != agent:
                        continue
                    total += 1
                    agreed = (view.get("direction") == trade["direction"]
                              or view.get("direction") == "neutral")
                    if (agreed and trade_correct) or (not agreed and not trade_correct):
                        correct += 1

            if total >= MIN_TRADES_FOR_WEIGHT:
                accuracy = correct / total
                # Linear map: accuracy=0 → 0.5, accuracy=0.5 → 1.0, accuracy=1.0 → 1.5
                self._weights[agent] = round(max(0.5, min(1.5, 0.5 + accuracy)), 3)
                log.info(f"FeedbackEngine: {agent} weight → {self._weights[agent]} "
                         f"(accuracy={accuracy:.1%} over {total} trades)")
            else:
                self._weights[agent] = DEFAULT_WEIGHT
