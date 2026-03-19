"""
agents/orchestrator.py
──────────────────────
3-Layer Multi-Agent Orchestrator

Layer 1 — Signal Generation (7 agents, parallel):
  momentum      — technical pattern anchor (veto power, provides entry/stop/target)
  fundamental   — earnings, valuation, analyst targets
  sentiment     — news NLP + price trend
  quant         — statistical extremes, realized vol
  macro         — VIX, yield curve, DXY, sector rotation
  options_flow  — put/call ratio, IV skew, unusual activity
  stat_arb      — pair spread Z-score, mean reversion

Layer 2 — Risk & Consensus (sequential):
  RegimeDetectionAgent  → classifies market (trending/ranging/volatile)
  ArbitrationAgent      → weighted voting with regime-adjusted weights
  PortfolioRiskAgent    → open position count, drawdown, correlation limits
  PositionSizingAgent   → Kelly-fraction sizing scaled by confidence + regime

Layer 3 — Execution Gate:
  ComplianceAgent       → hard rules: min equity, daily loss stop, PDT proxy

BASE_WEIGHTS is defined in agents/arbitration.py and re-exported here
so api/main.py can continue to reference _orch_module.BASE_WEIGHTS.
"""
import asyncio
import logging
from datetime import datetime

from agents.base          import AgentView
from agents.momentum      import MomentumAgent
from agents.sentiment     import SentimentAgent
from agents.fundamental   import FundamentalAgent
from agents.quant         import QuantAgent
from agents.macro         import MacroAgent
from agents.options_flow  import OptionsFlowAgent
from agents.stat_arb      import StatArbAgent

from agents.regime        import RegimeDetectionAgent, RegimeView
from agents.arbitration   import ArbitrationAgent, ArbitrationVerdict, BASE_WEIGHTS
from agents.portfolio_risk   import PortfolioRiskAgent
from agents.position_sizing  import PositionSizingAgent
from agents.compliance    import ComplianceAgent

from agents.feedback      import FeedbackEngine
from signals.engine       import TradeAlert

log = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, feedback_engine: FeedbackEngine):
        # ── Layer 1: Signal agents ─────────────────────────────────
        self.momentum     = MomentumAgent()
        self.sentiment    = SentimentAgent()
        self.fundamental  = FundamentalAgent()
        self.quant        = QuantAgent()
        self.macro        = MacroAgent()
        self.options_flow = OptionsFlowAgent()
        self.stat_arb     = StatArbAgent()

        # ── Layer 2: Risk & Consensus ──────────────────────────────
        self.regime          = RegimeDetectionAgent()
        self.arbitration     = ArbitrationAgent()
        self.portfolio_risk  = PortfolioRiskAgent()
        self.position_sizing = PositionSizingAgent()

        # ── Layer 3: Compliance ────────────────────────────────────
        self.compliance = ComplianceAgent()

        self.feedback = feedback_engine

        # Cache of the most-recent agent views for every symbol evaluated,
        # populated before any gates so non-signal symbols are still visible.
        self._views_cache: dict[str, list] = {}

    async def evaluate(
        self,
        symbol:         str,
        candles_5m:     "pd.DataFrame | None",
        candles_1h:     "pd.DataFrame | None",
        candles_1d:     "pd.DataFrame | None",
        quote:          "dict | None",
        news:           list[dict],
        positions:      list[dict]   = None,
        account_equity: float        = 0.0,
        daily_pl_pct:   float        = 0.0,
        traded_today:   "set | None" = None,
    ) -> "TradeAlert | None":
        """
        Run all 3 layers and return an enriched TradeAlert, or None if
        no high-conviction setup passes all gates.
        """
        if positions is None:
            positions = []
        if traded_today is None:
            traded_today = set()

        # ── LAYER 1: Run all 7 signal agents in parallel ──────────
        (m_view, s_view, f_view, q_view,
         mac_view, opt_view, sarb_view) = await asyncio.gather(
            asyncio.to_thread(self.momentum.analyze,    symbol, candles_5m, candles_1h),
            asyncio.to_thread(self.sentiment.analyze,   symbol, news, quote, candles_1d),
            asyncio.to_thread(self.fundamental.analyze, symbol),
            asyncio.to_thread(self.quant.analyze,       symbol, candles_1d, candles_5m),
            asyncio.to_thread(self.macro.analyze,       symbol),
            asyncio.to_thread(self.options_flow.analyze, symbol, quote),
            asyncio.to_thread(self.stat_arb.analyze,    symbol, candles_1d),
        )
        signal_views: list[AgentView] = [
            m_view, s_view, f_view, q_view, mac_view, opt_view, sarb_view
        ]

        # Always cache views so the UI can show agent analysis for every
        # scanned symbol — not just the ones that pass all gates.
        self._views_cache[symbol] = [v.to_dict() for v in signal_views]

        log.debug(
            f"{symbol}: L1 — momentum={m_view.direction} {m_view.confidence:.0%} | "
            f"sentiment={s_view.direction} {s_view.confidence:.0%} | "
            f"fundamental={f_view.direction} {f_view.confidence:.0%} | "
            f"quant={q_view.direction} {q_view.confidence:.0%} | "
            f"macro={mac_view.direction} {mac_view.confidence:.0%} | "
            f"options={opt_view.direction} {opt_view.confidence:.0%} | "
            f"stat_arb={sarb_view.direction} {sarb_view.confidence:.0%}"
        )

        # ── LAYER 2a: Regime detection (cached, cheap) ────────────
        regime: RegimeView = await asyncio.to_thread(
            self.regime.classify, candles_1d, quote
        )

        # ── LAYER 2b: Arbitration (weighted consensus) ────────────
        fb_weights = self.feedback.get_weights()
        verdict: ArbitrationVerdict = await asyncio.to_thread(
            self.arbitration.vote,
            signal_views, regime, BASE_WEIGHTS, fb_weights,
        )

        if not verdict.approved:
            log.debug(f"{symbol}: arbitration rejected — {verdict.rejection_reason}")
            return None

        # ── LAYER 2c: Portfolio risk gate ─────────────────────────
        risk_result = await asyncio.to_thread(
            self.portfolio_risk.evaluate,
            symbol, positions, account_equity, daily_pl_pct,
        )
        if not risk_result.approved:
            log.info(f"{symbol}: portfolio risk blocked — {risk_result.rejection_reason}")
            return None

        # ── LAYER 2d: Position sizing ─────────────────────────────
        win_rate, avg_win, avg_loss = self._feedback_sizing_stats()
        atr_pct = q_view.data.get("atr_pct", 0.02) or 0.02
        sizing = await asyncio.to_thread(
            self.position_sizing.calculate,
            account_equity or 10_000.0,
            verdict.consensus_score,
            win_rate, avg_win, avg_loss,
            regime.label,
            atr_pct,
        )

        # ── LAYER 3: Compliance hard gate ─────────────────────────
        compliance = await asyncio.to_thread(
            self.compliance.check,
            symbol, verdict.direction,
            positions, account_equity, daily_pl_pct, traded_today,
        )
        if not compliance.approved:
            log.info(f"{symbol}: compliance blocked — {compliance.rejection_reason}")
            return None

        # ── Build enriched TradeAlert ─────────────────────────────
        combined_signals = list(m_view.reasons)
        for view in [s_view, f_view, q_view, mac_view, opt_view, sarb_view]:
            if view.direction == verdict.direction or view.direction == "neutral":
                for r in view.reasons[:2]:
                    combined_signals.append(f"[{view.agent}] {r}")

        if verdict.opposing_agents:
            combined_signals.append(f"⚠ Opposing: {', '.join(verdict.opposing_agents)}")
        combined_signals.append(f"Regime: {regime.label} (VIX={regime.vix:.1f})")
        combined_signals.append(
            f"Risk sizing: {sizing.risk_fraction*100:.1f}% of equity"
            f" (Kelly={sizing.half_kelly:.3f})"
        )

        rr = round(
            abs(verdict.target - verdict.entry) / abs(verdict.entry - verdict.stop), 2
        ) if abs(verdict.entry - verdict.stop) > 0 else 0

        return TradeAlert(
            symbol          = symbol,
            setup           = verdict.setup,
            direction       = verdict.direction,
            entry           = verdict.entry,
            stop            = verdict.stop,
            target          = verdict.target,
            confidence      = verdict.consensus_score,
            risk_reward     = rr,
            timeframe       = verdict.timeframe,
            signals         = combined_signals,
            signal_id       = verdict.signal_id,
            agent_views     = [v.to_dict() for v in signal_views],
            consensus_score = verdict.consensus_score,
            agent_weights   = {k: round(v, 3)
                               for k, v in verdict.effective_weights.items()},
        )

    def _feedback_sizing_stats(self) -> tuple[float, float, float]:
        """Extract win_rate, avg_win_pct, avg_loss_pct from trade history."""
        history = self.feedback.get_history(n=50)
        if not history:
            return 0.50, 0.05, 0.02
        wins   = [t for t in history if t.get("outcome") == "win"]
        losses = [t for t in history if t.get("outcome") == "loss"]
        win_rate = len(wins) / len(history)
        avg_win  = (sum(t.get("pnl_pct", 0) or 0 for t in wins) / len(wins) / 100
                    if wins else 0.05)
        avg_loss = (abs(sum(t.get("pnl_pct", 0) or 0 for t in losses)) / len(losses) / 100
                    if losses else 0.02)
        return win_rate, max(avg_win, 0.001), max(avg_loss, 0.001)

    def get_last_regime(self) -> "RegimeView | None":
        """Returns the cached regime view (for /api/agents/status)."""
        return RegimeDetectionAgent._cache

    def get_agent_views_for_symbol(
        self,
        symbol, candles_5m, candles_1h, candles_1d, quote, news,
    ) -> list[dict]:
        """
        Synchronous wrapper — runs all 7 signal agents and returns their
        raw views. Used by /api/agents/analyze/{symbol} for on-demand
        analysis without requiring consensus to be met.
        """
        loop = asyncio.new_event_loop()
        try:
            views = loop.run_until_complete(asyncio.gather(
                asyncio.to_thread(self.momentum.analyze,     symbol, candles_5m, candles_1h),
                asyncio.to_thread(self.sentiment.analyze,    symbol, news or [], quote, candles_1d),
                asyncio.to_thread(self.fundamental.analyze,  symbol),
                asyncio.to_thread(self.quant.analyze,        symbol, candles_1d, candles_5m),
                asyncio.to_thread(self.macro.analyze,        symbol),
                asyncio.to_thread(self.options_flow.analyze, symbol, quote),
                asyncio.to_thread(self.stat_arb.analyze,     symbol, candles_1d),
            ))
            return [v.to_dict() for v in views]
        finally:
            loop.close()
