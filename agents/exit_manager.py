"""
agents/exit_manager.py
──────────────────────
ExitManager — tiered profit-taking logic for options positions.

Scale-out schedule (applied to every new position):
  Scale 1 — 50% of contracts — trigger at 30-40% P&L
             Earlier trigger (30%) if: RSI > 68 (call) / < 32 (put)
             OR MACD histogram declining from peak
             Later trigger (40%) if momentum still strong
  Scale 2 — 25% of contracts — trigger at 55-70% P&L
             Earlier if momentum fading (MACD bearish cross)
  Scale 3 — 25% of contracts — trigger at 100-120% P&L
             OR underlying reaches original TradeAlert target price

Roll evaluation (after scale 3 is hit):
  If agent conviction >= 0.58 AND momentum still aligned:
    Roll with 20-35% of scale3 proceeds (scaled by conviction)
    into next-expiry same strike

All thresholds are P&L on the OPTION (not the underlying).
"""
import math
import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ScaleLevel:
    qty:         int          # contracts to sell at this level
    min_pnl_pct: float        # lower bound of trigger range (e.g. 0.30)
    max_pnl_pct: float        # hard cap — always sell if reached (e.g. 0.40)
    target_price: float       # option premium target price
    filled:      bool = False
    filled_at:   str  = ""    # ISO timestamp when filled


@dataclass
class ScaleTargets:
    option_symbol:  str
    underlying:     str
    direction:      str        # "call" | "put"
    original_qty:   int
    cost_basis:     float      # premium paid per contract
    alert_target:   float      # underlying price target from TradeAlert
    scale1:         ScaleLevel = field(default_factory=lambda: ScaleLevel(0,0,0,0))
    scale2:         ScaleLevel = field(default_factory=lambda: ScaleLevel(0,0,0,0))
    scale3:         ScaleLevel = field(default_factory=lambda: ScaleLevel(0,0,0,0))
    roll_evaluated: bool = False


@dataclass
class RollSpec:
    should_roll:     bool
    roll_dollars:    float    # how much to put into the roll
    roll_fraction:   float    # fraction of scale3 proceeds used
    conviction:      float    # avg agent confidence that triggered roll
    rationale:       str


class ExitManager:

    def compute_scales(
        self,
        option_symbol: str,
        underlying:    str,
        direction:     str,
        cost_basis:    float,   # option price when bought (per contract)
        qty:           int,
        alert_target:  float,   # underlying target price from TradeAlert
    ) -> ScaleTargets:
        """
        Compute the three scale-out levels for a new position.
        qty splits 50% / 25% / 25% (rounding residual to scale3).
        """
        if qty <= 0 or cost_basis <= 0:
            return ScaleTargets(option_symbol, underlying, direction, qty, cost_basis, alert_target)

        s1_qty = math.ceil(qty * 0.50)
        s2_qty = math.ceil(qty * 0.25)
        s3_qty = qty - s1_qty - s2_qty   # absorbs rounding

        return ScaleTargets(
            option_symbol = option_symbol,
            underlying    = underlying,
            direction     = direction,
            original_qty  = qty,
            cost_basis    = cost_basis,
            alert_target  = alert_target,
            scale1 = ScaleLevel(
                qty          = s1_qty,
                min_pnl_pct  = 0.30,
                max_pnl_pct  = 0.40,
                target_price = round(cost_basis * 1.35, 2),
            ),
            scale2 = ScaleLevel(
                qty          = s2_qty,
                min_pnl_pct  = 0.55,
                max_pnl_pct  = 0.70,
                target_price = round(cost_basis * 1.625, 2),
            ),
            scale3 = ScaleLevel(
                qty          = max(s3_qty, 1) if qty >= 3 else 0,
                min_pnl_pct  = 1.00,
                max_pnl_pct  = 1.20,
                target_price = round(cost_basis * 2.10, 2),
            ),
        )

    def should_trigger_scale(
        self,
        scale:         ScaleLevel,
        current_pnl_pct: float,    # (current_price - cost_basis) / cost_basis
        candles_5m,                # pd.DataFrame or None (underlying 5m candles)
        quote:         dict,       # underlying quote
        direction:     str,
        underlying_price: float,
        alert_target:  float,
    ) -> tuple[bool, str]:
        """
        Returns (trigger, reason) for one scale level.

        Decision logic:
          - Below min_pnl_pct  → no trigger
          - Above max_pnl_pct  → always trigger (hard cap)
          - In soft zone       → check momentum + underlying level
            - RSI overbought/oversold → trigger at lower end
            - MACD hist declining    → trigger at lower end
            - Underlying at target   → trigger
            - Otherwise wait for upper end
        """
        if scale.filled:
            return False, "already filled"

        if current_pnl_pct < scale.min_pnl_pct:
            return False, f"P&L {current_pnl_pct:.0%} < min {scale.min_pnl_pct:.0%}"

        if current_pnl_pct >= scale.max_pnl_pct:
            return True, f"hard cap {scale.max_pnl_pct:.0%} reached ({current_pnl_pct:.0%})"

        # Check underlying target reached
        if direction == "call" and underlying_price >= alert_target * 0.98:
            return True, f"underlying near target (${underlying_price:.2f} ≥ ${alert_target:.2f})"
        if direction == "put" and underlying_price <= alert_target * 1.02:
            return True, f"underlying near target (${underlying_price:.2f} ≤ ${alert_target:.2f})"

        # Momentum check — early exit if fading
        if candles_5m is not None and len(candles_5m) >= 14:
            try:
                from agents.indicators import rsi, macd
                closes = (candles_5m["close"]
                          if "close" in candles_5m.columns
                          else candles_5m["Close"])
                rsi_val = float(rsi(closes).iloc[-1])
                _, _, hist = macd(closes)
                hist_last = float(hist.iloc[-1])
                hist_prev = float(hist.iloc[-2])

                if direction == "call":
                    if rsi_val > 68:
                        return True, f"RSI overbought ({rsi_val:.0f}) — early scale"
                    if hist_last < hist_prev < 0:
                        return True, f"MACD hist falling bearish — momentum fading"
                else:
                    if rsi_val < 32:
                        return True, f"RSI oversold ({rsi_val:.0f}) — early scale"
                    if hist_last > hist_prev > 0:
                        return True, f"MACD hist rising bullish — momentum fading"
            except Exception as e:
                log.debug(f"Momentum check error: {e}")

        # Still in soft zone, no exit signal — wait for upper end
        return False, f"in soft zone {current_pnl_pct:.0%}, waiting for {scale.max_pnl_pct:.0%}"

    def evaluate_roll(
        self,
        symbol:          str,
        agent_views:     list[dict],
        account_equity:  float,
        scale3_proceeds: float,    # dollars realized from scale3 sale
    ) -> RollSpec:
        """
        Decide whether to roll remaining profits into the next expiry.

        Roll conditions:
          1. At least 4 agent views available
          2. avg conviction >= 0.58 among views agreeing with original direction
          3. Roll size = 20-35% of scale3_proceeds, scaled by conviction
             capped at 2% of account_equity
        """
        if not agent_views or scale3_proceeds <= 0:
            return RollSpec(False, 0, 0, 0, "No views or zero proceeds")

        # Filter views that have a directional opinion
        directional = [v for v in agent_views if v.get("direction") != "neutral"]
        if len(directional) < 3:
            return RollSpec(False, 0, 0, 0, "Insufficient directional views for roll")

        # Majority direction
        calls = sum(1 for v in directional if v.get("direction") == "call")
        puts  = sum(1 for v in directional if v.get("direction") == "put")
        majority = "call" if calls >= puts else "put"

        aligned = [v for v in agent_views if v.get("direction") == majority]
        conviction = sum(v.get("confidence", 0) for v in aligned) / len(aligned) if aligned else 0

        if conviction < 0.58:
            return RollSpec(False, 0, 0, conviction,
                            f"Conviction {conviction:.0%} < 58% threshold")

        # Scale roll fraction: 20% at 58% conviction → 35% at 80%+ conviction
        roll_fraction = 0.20 + min(conviction - 0.58, 0.22) * (0.15 / 0.22)
        roll_dollars  = min(scale3_proceeds * roll_fraction,
                           account_equity * 0.02)
        roll_dollars  = round(roll_dollars, 2)

        return RollSpec(
            should_roll   = True,
            roll_dollars  = roll_dollars,
            roll_fraction = round(roll_fraction, 3),
            conviction    = round(conviction, 3),
            rationale     = (f"Conviction {conviction:.0%} across {len(aligned)} agents "
                             f"→ rolling {roll_fraction:.0%} of scale3 proceeds "
                             f"(${roll_dollars:.2f})"),
        )

    def to_dict(self, st: ScaleTargets) -> dict:
        """Serialize ScaleTargets for state storage and API responses."""
        def _sl(sl: ScaleLevel) -> dict:
            return {
                "qty": sl.qty, "min_pnl_pct": sl.min_pnl_pct,
                "max_pnl_pct": sl.max_pnl_pct, "target_price": sl.target_price,
                "filled": sl.filled, "filled_at": sl.filled_at,
            }
        return {
            "option_symbol": st.option_symbol, "underlying": st.underlying,
            "direction": st.direction, "original_qty": st.original_qty,
            "cost_basis": st.cost_basis, "alert_target": st.alert_target,
            "scale1": _sl(st.scale1), "scale2": _sl(st.scale2), "scale3": _sl(st.scale3),
            "roll_evaluated": st.roll_evaluated,
        }
