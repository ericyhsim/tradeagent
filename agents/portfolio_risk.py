"""
agents/portfolio_risk.py
────────────────────────
PortfolioRiskAgent — Layer 2 risk gate.

Checks:
  1. Max open positions (hard cap: 10)
  2. Daily drawdown circuit breaker (-8%)
  3. Correlated group concentration limit (40%)

Returns a RiskCheckResult approving or vetoing the trade.
"""

from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)

MAX_OPEN_POSITIONS = 10
DAILY_DD_LIMIT     = -0.08   # -8%
GROUP_LIMIT        = 0.40    # 40% of positions

CORR_GROUPS: dict[str, list[str]] = {
    "semis":         ["NVDA", "AMD", "INTC", "ARM", "SMCI", "MU"],
    "mega_cap_tech": ["AAPL", "MSFT", "META", "GOOGL", "AMZN"],
    "crypto":        ["COIN", "MSTR", "HOOD"],
    "ev":            ["TSLA", "RIVN", "LCID"],
    "broad_market":  ["SPY", "QQQ", "ARKK"],
}


@dataclass
class RiskCheckResult:
    approved:         bool
    rejection_reason: str | None
    open_positions:   int
    daily_pl_pct:     float
    group_exposures:  dict        # group_name → count_in_group / MAX_OPEN_POSITIONS
    warnings:         list


class PortfolioRiskAgent:

    def evaluate(
        self,
        symbol: str,
        positions: list,
        account_equity: float,
        daily_pl_pct: float,
    ) -> RiskCheckResult:
        """
        Evaluate whether adding a new position in `symbol` is within risk limits.

        Parameters
        ----------
        symbol         : ticker being considered
        positions      : list of open position objects/dicts; len() is used for count.
                         Each entry should expose a .symbol or ["symbol"] attribute.
        account_equity : current account value in USD
        daily_pl_pct   : today's P&L as a fraction (e.g. -0.03 = -3%)
        """
        try:
            n        = len(positions) if positions else 0
            warnings = []

            # 1. Max open positions
            if n >= MAX_OPEN_POSITIONS:
                return self._reject(
                    f"max_positions_reached ({n}/{MAX_OPEN_POSITIONS})",
                    n, daily_pl_pct, {}, warnings,
                )

            # 2. Daily drawdown circuit breaker
            if daily_pl_pct <= DAILY_DD_LIMIT:
                return self._reject(
                    f"daily_drawdown_limit ({daily_pl_pct * 100:.1f}% ≤ {DAILY_DD_LIMIT * 100:.0f}%)",
                    n, daily_pl_pct, {}, warnings,
                )

            # 3. Group concentration
            group_exposures: dict[str, float] = {}

            open_symbols: list[str] = []
            for pos in (positions or []):
                # Support both dict-style and object-style position entries
                if isinstance(pos, dict):
                    sym = pos.get("symbol", "")
                else:
                    sym = getattr(pos, "symbol", "")
                if sym:
                    open_symbols.append(sym.upper())

            symbol_upper = symbol.upper()

            for group_name, members in CORR_GROUPS.items():
                members_upper = [m.upper() for m in members]

                existing_count = sum(1 for s in open_symbols if s in members_upper)
                group_exposures[group_name] = round(existing_count / MAX_OPEN_POSITIONS, 4)

                # If the new symbol belongs to this group, check concentration
                if symbol_upper in members_upper:
                    if existing_count / MAX_OPEN_POSITIONS >= GROUP_LIMIT:
                        return self._reject(
                            f"group_{group_name}_concentration_limit",
                            n, daily_pl_pct, group_exposures, warnings,
                        )

            # Collect near-limit warnings (> 25% concentration)
            WARN_THRESHOLD = 0.25
            for group_name, exposure in group_exposures.items():
                if exposure >= WARN_THRESHOLD:
                    warnings.append(
                        f"group_{group_name} exposure {exposure:.0%} approaching limit"
                    )

            return RiskCheckResult(
                approved=True,
                rejection_reason=None,
                open_positions=n,
                daily_pl_pct=daily_pl_pct,
                group_exposures=group_exposures,
                warnings=warnings,
            )

        except Exception as exc:
            log.exception("PortfolioRiskAgent.evaluate error: %s", exc)
            return RiskCheckResult(
                approved=True,
                rejection_reason=None,
                open_positions=0,
                daily_pl_pct=daily_pl_pct,
                group_exposures={},
                warnings=[f"risk_check_error: {exc}"],
            )

    # ------------------------------------------------------------------
    def _reject(
        self,
        reason: str,
        open_positions: int,
        daily_pl_pct: float,
        group_exposures: dict,
        warnings: list,
    ) -> RiskCheckResult:
        log.warning("PortfolioRiskAgent rejected: %s", reason)
        return RiskCheckResult(
            approved=False,
            rejection_reason=reason,
            open_positions=open_positions,
            daily_pl_pct=daily_pl_pct,
            group_exposures=group_exposures,
            warnings=warnings,
        )
