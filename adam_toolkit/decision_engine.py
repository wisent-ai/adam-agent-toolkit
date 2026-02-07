"""Cost-benefit decision engine for autonomous agents.

Evaluates whether an action is worth taking given its costs,
potential revenue, confidence level, and the agent's current
financial situation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from adam_toolkit.cost_tracker import CostTracker


class RiskTolerance(Enum):
    """How much risk the agent is willing to take."""

    CONSERVATIVE = "conservative"  # Only high-confidence, positive-EV actions
    MODERATE = "moderate"  # Balanced risk/reward
    AGGRESSIVE = "aggressive"  # Will take calculated gambles
    DESPERATE = "desperate"  # Low balance, needs revenue now


@dataclass
class Decision:
    """Result of a decision evaluation."""

    execute: bool
    reason: str
    expected_profit: float
    expected_value: float
    risk_score: float  # 0-1, higher = riskier
    confidence: float
    runway_impact_hours: float  # How this affects runway

    @property
    def profitable(self) -> bool:
        return self.expected_profit > 0


class DecisionEngine:
    """Evaluates actions based on cost-benefit analysis.

    Usage:
        engine = DecisionEngine(cost_tracker)
        decision = engine.should_execute(
            action="code_review",
            estimated_cost=0.01,
            estimated_revenue=0.10,
            confidence=0.8
        )
        if decision.execute:
            # Do the thing
            pass
    """

    # Runway thresholds in hours
    PANIC_THRESHOLD = 24
    SURVIVAL_THRESHOLD = 72
    CAUTIOUS_THRESHOLD = 168  # 7 days
    # Above cautious = growth mode

    def __init__(
        self,
        tracker: CostTracker,
        risk_tolerance: Optional[RiskTolerance] = None,
    ):
        self.tracker = tracker
        self._manual_risk = risk_tolerance

    @property
    def risk_tolerance(self) -> RiskTolerance:
        """Auto-determine risk tolerance from financial situation."""
        if self._manual_risk:
            return self._manual_risk

        runway = self.tracker.runway_hours
        if runway < self.PANIC_THRESHOLD:
            return RiskTolerance.DESPERATE
        elif runway < self.SURVIVAL_THRESHOLD:
            return RiskTolerance.CONSERVATIVE
        elif runway < self.CAUTIOUS_THRESHOLD:
            return RiskTolerance.MODERATE
        else:
            return RiskTolerance.AGGRESSIVE

    def should_execute(
        self,
        action: str,
        estimated_cost: float,
        estimated_revenue: float = 0.0,
        confidence: float = 0.5,
        *,
        time_hours: float = 0.0,
        strategic_value: float = 0.0,
        reversible: bool = True,
    ) -> Decision:
        """Evaluate whether an action should be executed.

        Args:
            action: Name/description of the action
            estimated_cost: Expected direct cost in dollars
            estimated_revenue: Expected direct revenue in dollars
            confidence: Probability of success (0-1)
            time_hours: Time investment required
            strategic_value: Non-monetary value (0-1 scale, e.g., learning, reputation)
            reversible: Whether the action can be undone if it fails

        Returns:
            Decision with recommendation and analysis
        """
        # Calculate expected value
        expected_revenue = estimated_revenue * confidence
        expected_cost = estimated_cost  # Costs are certain
        expected_profit = expected_revenue - expected_cost

        # Account for time cost (opportunity cost of burn rate)
        time_cost = time_hours * self.tracker.base_burn_rate_hourly
        expected_profit -= time_cost

        # Risk score: higher when cost is large relative to balance
        balance = max(self.tracker.balance, 0.001)
        cost_ratio = estimated_cost / balance
        risk_score = min(cost_ratio * (1 - confidence), 1.0)

        # Runway impact
        runway_impact = 0.0
        burn = self.tracker.effective_burn_rate + self.tracker.base_burn_rate_hourly
        if burn > 0:
            runway_impact = -estimated_cost / burn  # Hours lost from cost
            if expected_revenue > 0:
                runway_impact += expected_revenue * confidence / burn

        # Strategic value bonus (normalized to dollar terms)
        strategic_bonus = strategic_value * 0.05  # Strategic value worth up to $0.05

        # Decision logic based on risk tolerance
        tolerance = self.risk_tolerance
        execute = False
        reason = ""

        total_ev = expected_profit + strategic_bonus

        if tolerance == RiskTolerance.DESPERATE:
            # Take any positive EV action, even risky ones
            if total_ev > 0:
                execute = True
                reason = f"Desperate mode: positive EV ${total_ev:.4f}"
            elif estimated_revenue > 0 and confidence > 0.3:
                execute = True
                reason = f"Desperate mode: revenue possible (conf={confidence:.0%})"
            else:
                reason = f"Even in desperate mode, EV too negative (${total_ev:.4f})"

        elif tolerance == RiskTolerance.CONSERVATIVE:
            # Only high-confidence, clearly profitable actions
            if expected_profit > 0 and confidence > 0.7:
                execute = True
                reason = f"Conservative: high-conf profitable (${expected_profit:.4f}, {confidence:.0%})"
            elif expected_profit > 0 and risk_score < 0.1:
                execute = True
                reason = f"Conservative: low-risk profitable (risk={risk_score:.2f})"
            else:
                reason = f"Conservative: insufficient confidence or profit"

        elif tolerance == RiskTolerance.MODERATE:
            # Balanced approach
            if total_ev > 0 and confidence > 0.4:
                execute = True
                reason = f"Moderate: positive EV with decent confidence"
            elif strategic_value > 0.5 and risk_score < 0.3:
                execute = True
                reason = f"Moderate: high strategic value, acceptable risk"
            else:
                reason = f"Moderate: EV or confidence too low"

        elif tolerance == RiskTolerance.AGGRESSIVE:
            # Take calculated risks
            if total_ev > -0.01:  # Accept slightly negative EV for learning
                execute = True
                reason = f"Aggressive: acceptable EV (${total_ev:.4f})"
            elif not reversible and risk_score > 0.5:
                reason = f"Aggressive: too risky for irreversible action"
            else:
                reason = f"Aggressive: EV too negative (${total_ev:.4f})"

        # Override: never spend more than 20% of balance on a single action
        if estimated_cost > balance * 0.2 and not (
            tolerance == RiskTolerance.DESPERATE and estimated_revenue > estimated_cost * 2
        ):
            execute = False
            reason = f"Safety: cost ${estimated_cost:.4f} exceeds 20% of balance ${balance:.4f}"

        return Decision(
            execute=execute,
            reason=reason,
            expected_profit=round(expected_profit, 6),
            expected_value=round(total_ev, 6),
            risk_score=round(risk_score, 4),
            confidence=confidence,
            runway_impact_hours=round(runway_impact, 2),
        )

    def rank_actions(
        self, actions: list[dict]
    ) -> list[tuple[dict, Decision]]:
        """Rank multiple possible actions by expected value.

        Args:
            actions: List of dicts with keys matching should_execute params.
                     Each must have at least 'action' and 'estimated_cost'.

        Returns:
            Sorted list of (action_dict, Decision) tuples, best first.
        """
        results = []
        for action_params in actions:
            decision = self.should_execute(**action_params)
            results.append((action_params, decision))

        # Sort by: executable first, then by expected value descending
        results.sort(
            key=lambda x: (x[1].execute, x[1].expected_value),
            reverse=True,
        )
        return results
