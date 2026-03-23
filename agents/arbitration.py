"""
agents/arbitration.py
─────────────────────
ArbitrationAgent — collects all 7 signal views, applies regime-adjusted
weights, and issues a final ArbitrationVerdict.

The MomentumAgent is the anchor:
  • Must have direction != "neutral" and confidence >= MIN_ANCHOR_CONF
  • Provides entry / stop / target levels (from its data dict)
  • Provides R:R gate (>= 2.0)

BASE_WEIGHTS: the starting weight for each signal agent before
feedback and regime adjustments. Exported for use by api/main.py.
"""

from dataclasses import dataclass, field
from agents.base import AgentView
import logging
import uuid
from datetime import datetime

log = logging.getLogger(__name__)

MIN_ANCHOR_CONF = 0.76   # momentum anchor confidence floor
MIN_CONSENSUS   = 0.62   # calibrated for 3-agent denominator (momentum=5, quant=4.5, options_flow=2.5)
                         # requires momentum + quant agreement, or momentum + options_flow + neutral quant
MIN_RR          = 2.5

# Simplified 3-agent system:
#   momentum    — anchor + veto (must agree, provides entry/stop/target)
#   quant       — confirming veto (ATR, vol, statistical edge)
#   options_flow — confirming lift (smart money positioning)
#   All others zeroed out — they added noise without improving win rate.
BASE_WEIGHTS: dict[str, float] = {
    "momentum":     5.0,
    "fundamental":  0.0,   # removed — no predictive value on intraday setups
    "quant":        4.5,   # confirming veto
    "sentiment":    0.0,   # removed — too lagged for our timeframe
    "macro":        0.0,   # removed — macro regime handled by RegimeDetectionAgent
    "options_flow": 2.5,   # elevated — smart money flow is the strongest confirmer
    "stat_arb":     0.0,   # removed — near-zero historically, adds API overhead
}

# ── Regime presets ────────────────────────────────────────────────────────
# Each preset fully replaces BASE_WEIGHTS when applied.
# Rationale:
#   trending      — momentum + macro dominate; stat_arb muted (pairs diverge in trends)
#   ranging       — mean-reversion (stat_arb + quant) dominate; momentum muted
#   high_vol      — macro + options_flow dominate; stat_arb dangerous (correlations break)
#   crisis        — macro + options_flow at max; fundamentals irrelevant in panic

# Regime presets only adjust the 3 active agents.
# Zeroed agents stay at 0 regardless of regime.
REGIME_PRESETS: dict[str, dict[str, float]] = {
    "trending": {
        "momentum": 6.0, "fundamental": 0.0, "quant": 3.5,
        "sentiment": 0.0, "macro": 0.0, "options_flow": 2.5, "stat_arb": 0.0,
    },
    "ranging": {
        "momentum": 4.0, "fundamental": 0.0, "quant": 5.5,
        "sentiment": 0.0, "macro": 0.0, "options_flow": 2.5, "stat_arb": 0.0,
    },
    "high_volatility": {
        "momentum": 4.0, "fundamental": 0.0, "quant": 3.0,
        "sentiment": 0.0, "macro": 0.0, "options_flow": 5.0, "stat_arb": 0.0,
    },
    "crisis": {
        "momentum": 3.0, "fundamental": 0.0, "quant": 3.0,
        "sentiment": 0.0, "macro": 0.0, "options_flow": 6.0, "stat_arb": 0.0,
    },
}


def apply_preset(name: str) -> dict:
    """
    Apply a named preset by mutating BASE_WEIGHTS in-place.
    Returns the updated BASE_WEIGHTS dict.
    Raises KeyError if name is not a valid preset.
    """
    if name not in REGIME_PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Valid: {list(REGIME_PRESETS)}")
    BASE_WEIGHTS.update(REGIME_PRESETS[name])
    return dict(BASE_WEIGHTS)


@dataclass
class ArbitrationVerdict:
    approved:          bool
    direction:         str          # from momentum anchor
    consensus_score:   float
    effective_weights: dict         # agent → effective weight used
    supporting_agents: list
    opposing_agents:   list
    rejection_reason:  str | None   # None if approved
    signal_id:         str
    entry:             float | None
    stop:              float | None
    target:            float | None
    setup:             str
    timeframe:         str
    regime_label:      str


class ArbitrationAgent:

    def vote(
        self,
        signal_views: list,
        regime,
        base_weights: dict,
        feedback_weights: dict,
    ) -> ArbitrationVerdict:
        """
        Aggregate all agent views into a single ArbitrationVerdict.

        Parameters
        ----------
        signal_views    : list of AgentView from all signal agents
        regime          : RegimeView — carries .label and .weight_multipliers
        base_weights    : dict[str, float] — base weight per agent name
        feedback_weights: dict[str, float] — RL feedback multipliers per agent
        """
        views = {v.agent: v for v in signal_views}

        # --- Anchor check ---
        anchor = views.get("momentum")
        if anchor is None:
            return self._reject(
                "no_momentum_anchor", None, signal_views, regime.label
            )

        if anchor.direction == "neutral" or anchor.confidence < MIN_ANCHOR_CONF:
            return self._reject(
                f"momentum_gate: dir={anchor.direction} conf={anchor.confidence:.0%}",
                anchor.direction,
                signal_views,
                regime.label,
            )

        direction = anchor.direction

        # --- Extract price levels from anchor ---
        entry    = anchor.data.get("entry")
        stop     = anchor.data.get("stop")
        target   = anchor.data.get("target")
        setup    = anchor.data.get("setup", "multi_agent")
        timeframe = anchor.data.get("timeframe", "weekly")

        if not (entry and stop and target):
            return self._reject(
                "no_price_levels", direction, signal_views, regime.label
            )

        # --- R:R gate ---
        risk   = abs(entry - stop)
        reward = abs(target - entry)
        rr     = reward / risk if risk > 0 else 0.0

        if rr < MIN_RR:
            return self._reject(
                f"rr_{rr:.2f}_below_2.0", direction, signal_views, regime.label
            )

        # --- Weighted consensus ---
        numerator   = 0.0
        denominator = 0.0
        effective_weights: dict[str, float] = {}
        supporting_agents: list[str] = []
        opposing_agents:   list[str] = []

        for v in signal_views:
            base_w = base_weights.get(v.agent, 1.0)
            fb_w   = feedback_weights.get(v.agent, 1.0)
            reg_w  = regime.weight_multipliers.get(v.agent, 1.0)
            eff_w  = base_w * fb_w * reg_w

            effective_weights[v.agent] = round(eff_w, 4)

            agreed    = (v.direction == direction or v.direction == "neutral")
            eff_conf  = v.confidence if agreed else (1.0 - v.confidence)

            numerator   += eff_w * eff_conf
            denominator += eff_w

            # Exclude anchor from supporting/opposing lists for clarity
            if v.agent != "momentum":
                if agreed:
                    supporting_agents.append(v.agent)
                else:
                    opposing_agents.append(v.agent)

        consensus = numerator / denominator if denominator else 0.0

        if consensus < MIN_CONSENSUS:
            return self._reject(
                f"consensus_{consensus:.2f}_below_{MIN_CONSENSUS}",
                direction,
                signal_views,
                regime.label,
            )

        signal_id = (
            f"{anchor.agent.upper()}-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-"
            f"{uuid.uuid4().hex[:4]}"
        )

        return ArbitrationVerdict(
            approved=True,
            direction=direction,
            consensus_score=round(consensus, 4),
            effective_weights=effective_weights,
            supporting_agents=supporting_agents,
            opposing_agents=opposing_agents,
            rejection_reason=None,
            signal_id=signal_id,
            entry=entry,
            stop=stop,
            target=target,
            setup=setup,
            timeframe=timeframe,
            regime_label=regime.label,
        )

    # ------------------------------------------------------------------
    def _reject(
        self,
        reason: str,
        anchor_dir: str | None,
        signal_views: list,
        regime_label: str,
    ) -> ArbitrationVerdict:
        log.info("ArbitrationAgent rejected: %s", reason)
        return ArbitrationVerdict(
            approved=False,
            direction=anchor_dir or "neutral",
            consensus_score=0.0,
            effective_weights={},
            supporting_agents=[],
            opposing_agents=[],
            rejection_reason=reason,
            signal_id="",
            entry=None,
            stop=None,
            target=None,
            setup="rejected",
            timeframe="weekly",
            regime_label=regime_label,
        )
