"""
agents/position_sizing.py
─────────────────────────
PositionSizingAgent — Kelly criterion + volatility-scaled sizing.

Formula: risk_fraction = half_kelly * confidence_scale * regime_scale * vol_scale
Clamped to [0.5%, 2.5%] of account equity.

Inputs: feedback win-rate statistics + regime label + ATR%.
"""

from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)

MIN_RISK_FRACTION = 0.005
MAX_RISK_FRACTION = 0.025
DEFAULT_HALF_KELLY = 0.01

REGIME_SCALE: dict[str, float] = {
    "trending_up":   1.0,
    "trending_down": 1.0,
    "ranging":       0.8,
    "volatile":      0.6,
}


@dataclass
class SizingSpec:
    risk_fraction:    float
    risk_dollars:     float
    half_kelly:       float
    confidence_scale: float
    regime_scale:     float
    vol_scale:        float
    rationale:        str


class PositionSizingAgent:

    def calculate(
        self,
        account_equity: float,
        confidence: float,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        regime_label: str,
        atr_pct: float,
    ) -> SizingSpec:
        """
        Compute the risk fraction to allocate to a new trade.

        Parameters
        ----------
        account_equity : total account value in USD
        confidence     : signal confidence 0.0–1.0
        win_rate       : historical win rate 0.0–1.0 (from feedback stats)
        avg_win_pct    : average winning trade return as a fraction
        avg_loss_pct   : average losing trade loss as a positive fraction
        regime_label   : one of "trending_up" | "trending_down" | "ranging" | "volatile"
        atr_pct        : ATR as a fraction of price (e.g. 0.03 = 3%)
        """
        try:
            # --- Kelly criterion ---
            if win_rate > 0 and avg_win_pct > 0 and avg_loss_pct > 0:
                b          = avg_win_pct / avg_loss_pct
                kelly      = (win_rate * b - (1 - win_rate)) / b
                half_kelly = max(0.0, kelly * 0.5)
            else:
                half_kelly = DEFAULT_HALF_KELLY

            # --- Scale factors ---
            confidence_scale = max(0.3, min(confidence, 1.0))
            regime_scale_val = REGIME_SCALE.get(regime_label, 0.8)

            if atr_pct > 0.04:
                vol_scale = 0.6
            elif atr_pct > 0.025:
                vol_scale = 0.8
            else:
                vol_scale = 1.0

            # --- Final risk fraction ---
            risk_fraction = half_kelly * confidence_scale * regime_scale_val * vol_scale
            risk_fraction = max(MIN_RISK_FRACTION, min(risk_fraction, MAX_RISK_FRACTION))

            risk_dollars = round(account_equity * risk_fraction, 2) if account_equity > 0 else 0.0

            rationale = (
                f"½Kelly={half_kelly:.3f} × conf={confidence_scale:.2f} "
                f"× regime={regime_scale_val:.1f} × vol={vol_scale:.1f} "
                f"→ {risk_fraction:.2%}"
            )

            return SizingSpec(
                risk_fraction=round(risk_fraction, 6),
                risk_dollars=risk_dollars,
                half_kelly=round(half_kelly, 6),
                confidence_scale=round(confidence_scale, 4),
                regime_scale=regime_scale_val,
                vol_scale=vol_scale,
                rationale=rationale,
            )

        except Exception as exc:
            log.exception("PositionSizingAgent.calculate error: %s", exc)
            risk_dollars = round(account_equity * MIN_RISK_FRACTION, 2) if account_equity > 0 else 0.0
            return SizingSpec(
                risk_fraction=MIN_RISK_FRACTION,
                risk_dollars=risk_dollars,
                half_kelly=DEFAULT_HALF_KELLY,
                confidence_scale=0.3,
                regime_scale=0.8,
                vol_scale=1.0,
                rationale=f"fallback_min_size (error: {exc})",
            )
