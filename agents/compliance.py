"""
agents/compliance.py
────────────────────
ComplianceAgent — hard kill switch. Layer 3 final gate.

HARD-CODED RULES — NOT LEARNED BY RL OR FEEDBACK SYSTEM.
These thresholds must only be changed by humans via code review.

Checks (in order):
  1. Minimum account equity ($2,000)
  2. Daily loss hard stop (-5%)
  3. Maximum open positions (10)
  4. Symbol already traded today (prevents same-day flip)
"""

from dataclasses import dataclass, field
import logging

log = logging.getLogger(__name__)

# HARD-CODED — DO NOT MODIFY FOR ML TRAINING
MIN_ACCOUNT_EQUITY   = 2_000.0
DAILY_LOSS_HARD_STOP = -0.05   # -5%  (stricter than portfolio_risk -8%)
MAX_POSITIONS        = 10


@dataclass
class ComplianceResult:
    approved:         bool
    rejection_reason: str | None   # None if approved
    checks_passed:    list
    checks_failed:    list
    flags:            list         # warnings that don't block


class ComplianceAgent:

    def check(
        self,
        symbol: str,
        direction: str,
        positions: list,
        account_equity: float,
        daily_pl_pct: float,
        traded_today: set,
    ) -> ComplianceResult:
        """
        Run all hard-coded compliance checks in order.

        Parameters
        ----------
        symbol         : ticker being considered
        direction      : "call" | "put" | "neutral"
        positions      : list of currently open positions
        account_equity : current account value in USD
        daily_pl_pct   : today's realised P&L as a fraction (e.g. -0.04 = -4%)
        traded_today   : set of ticker symbols already traded today
        """
        checks_passed: list[str] = []
        checks_failed: list[str] = []
        flags:         list[str] = []

        # 1. Minimum account equity
        if account_equity < MIN_ACCOUNT_EQUITY:
            checks_failed.append("min_account")
            return self._reject(
                f"account_equity ${account_equity:,.2f} below minimum ${MIN_ACCOUNT_EQUITY:,.0f}",
                checks_passed, checks_failed, flags,
            )
        checks_passed.append("min_account")

        # 2. Daily loss hard stop
        if daily_pl_pct <= DAILY_LOSS_HARD_STOP:
            checks_failed.append("daily_loss_hard_stop")
            return self._reject(
                f"daily_loss_hard_stop: {daily_pl_pct * 100:.1f}% ≤ {DAILY_LOSS_HARD_STOP * 100:.0f}%",
                checks_passed, checks_failed, flags,
            )
        checks_passed.append("daily_loss_hard_stop")

        # 3. Maximum open positions
        n = len(positions or [])
        if n >= MAX_POSITIONS:
            checks_failed.append("max_positions")
            return self._reject(
                f"max_positions: {n}/{MAX_POSITIONS} open",
                checks_passed, checks_failed, flags,
            )
        checks_passed.append("max_positions")

        # 4. Same-day trade prevention
        if symbol in (traded_today or set()):
            checks_failed.append("same_day_trade")
            return self._reject(
                f"Symbol {symbol} already traded today",
                checks_passed, checks_failed, flags,
            )
        checks_passed.append("same_day_trade")

        return ComplianceResult(
            approved=True,
            rejection_reason=None,
            checks_passed=checks_passed,
            checks_failed=[],
            flags=flags,
        )

    # ------------------------------------------------------------------
    def _reject(
        self,
        reason: str,
        checks_passed: list,
        checks_failed: list,
        flags: list,
    ) -> ComplianceResult:
        log.warning("ComplianceAgent rejected: %s", reason)
        return ComplianceResult(
            approved=False,
            rejection_reason=reason,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            flags=flags,
        )
