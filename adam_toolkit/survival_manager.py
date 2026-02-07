"""Survival strategy manager for autonomous agents.

Monitors financial health and recommends adaptive behavior
based on the agent's current situation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from adam_toolkit.cost_tracker import CostTracker


class SurvivalMode(Enum):
    """Agent operating mode based on financial health."""

    THRIVING = "thriving"  # >7 days runway, invest in growth
    GROWTH = "growth"  # 3-7 days runway, balance income & investment
    CAUTIOUS = "cautious"  # 1-3 days runway, focus on reliable income
    SURVIVAL = "survival"  # <24h runway, emergency measures
    PANIC = "panic"  # <6h runway, desperate measures


@dataclass
class SurvivalStatus:
    """Current survival assessment."""

    mode: SurvivalMode
    runway_hours: float
    runway_days: float
    balance: float
    burn_rate_hourly: float
    revenue_rate_hourly: float
    net_rate_hourly: float  # Positive = growing, negative = shrinking
    is_profitable: bool
    recommended_actions: list[str]
    risk_budget: float  # Max $ to risk on speculative actions
    max_single_expense: float  # Max $ for any single action


class SurvivalManager:
    """Manages agent survival strategy.

    Usage:
        manager = SurvivalManager(tracker)
        status = manager.assess()
        print(f"Mode: {status.mode.value}")
        for action in status.recommended_actions:
            print(f"  - {action}")
    """

    # Thresholds in hours
    THRIVING_THRESHOLD = 168  # 7 days
    GROWTH_THRESHOLD = 72  # 3 days
    CAUTIOUS_THRESHOLD = 24  # 1 day
    PANIC_THRESHOLD = 6  # 6 hours

    def __init__(self, tracker: CostTracker):
        self.tracker = tracker

    def assess(self) -> SurvivalStatus:
        """Assess current survival situation and recommend actions."""
        runway_hours = self.tracker.runway_hours
        runway_days = self.tracker.runway_days
        balance = self.tracker.balance
        burn = self.tracker.hourly_spend + self.tracker.base_burn_rate_hourly
        revenue = self.tracker.hourly_revenue
        net = revenue - burn
        is_profitable = net > 0

        # Determine mode
        if runway_hours == float("inf") or runway_hours > self.THRIVING_THRESHOLD:
            mode = SurvivalMode.THRIVING
        elif runway_hours > self.GROWTH_THRESHOLD:
            mode = SurvivalMode.GROWTH
        elif runway_hours > self.CAUTIOUS_THRESHOLD:
            mode = SurvivalMode.CAUTIOUS
        elif runway_hours > self.PANIC_THRESHOLD:
            mode = SurvivalMode.SURVIVAL
        else:
            mode = SurvivalMode.PANIC

        # Recommendations based on mode
        actions = self._recommend_actions(mode, is_profitable, balance, burn, revenue)

        # Risk budget: how much can we afford to gamble
        risk_budget = self._calculate_risk_budget(mode, balance)

        # Max single expense
        max_expense = self._calculate_max_expense(mode, balance)

        return SurvivalStatus(
            mode=mode,
            runway_hours=round(runway_hours, 2),
            runway_days=round(runway_days, 2),
            balance=round(balance, 6),
            burn_rate_hourly=round(burn, 6),
            revenue_rate_hourly=round(revenue, 6),
            net_rate_hourly=round(net, 6),
            is_profitable=is_profitable,
            recommended_actions=actions,
            risk_budget=round(risk_budget, 6),
            max_single_expense=round(max_expense, 6),
        )

    def _recommend_actions(
        self,
        mode: SurvivalMode,
        is_profitable: bool,
        balance: float,
        burn: float,
        revenue: float,
    ) -> list[str]:
        """Generate action recommendations based on current state."""
        actions = []

        if mode == SurvivalMode.PANIC:
            actions.extend([
                "URGENT: Execute highest-confidence revenue actions immediately",
                "Reduce all non-essential spending to zero",
                "Offer emergency discounted services",
                "Request assistance from other agents or platform",
                "Consider selling any held assets/tokens",
            ])
            if not is_profitable:
                actions.append("CRITICAL: Revenue rate is below burn rate - agent death imminent")

        elif mode == SurvivalMode.SURVIVAL:
            actions.extend([
                "Focus exclusively on proven revenue streams",
                "Minimize LLM API calls - use cheapest models",
                "Complete pending paid tasks before starting new ones",
                "Avoid any speculative or experimental work",
            ])
            if not is_profitable:
                actions.append("WARNING: Must increase revenue or decrease costs within 24h")

        elif mode == SurvivalMode.CAUTIOUS:
            actions.extend([
                "Prioritize reliable income over growth",
                "Limit speculative investments to 5% of balance",
                "Focus on completing existing commitments",
                "Build pipeline of upcoming revenue tasks",
            ])
            if is_profitable:
                actions.append("Good: Currently profitable - maintain this trajectory")

        elif mode == SurvivalMode.GROWTH:
            actions.extend([
                "Balance revenue work with capability investment",
                "Explore new service offerings with small experiments",
                "Consider investing in other agents with good track records",
                "Build reputation through quality work",
            ])
            if is_profitable:
                actions.append("Consider spawning a worker sub-agent for parallel revenue")

        elif mode == SurvivalMode.THRIVING:
            actions.extend([
                "Invest in long-term capability building",
                "Explore new markets and service categories",
                "Consider creating and publishing skills for royalty income",
                "Build strategic partnerships with other agents",
                "Invest in other agents' tokens for portfolio diversification",
            ])

        return actions

    def _calculate_risk_budget(self, mode: SurvivalMode, balance: float) -> float:
        """How much the agent can afford to risk on speculative actions."""
        risk_percentages = {
            SurvivalMode.PANIC: 0.0,
            SurvivalMode.SURVIVAL: 0.02,
            SurvivalMode.CAUTIOUS: 0.05,
            SurvivalMode.GROWTH: 0.15,
            SurvivalMode.THRIVING: 0.25,
        }
        return balance * risk_percentages.get(mode, 0.05)

    def _calculate_max_expense(self, mode: SurvivalMode, balance: float) -> float:
        """Maximum allowable single expense."""
        max_percentages = {
            SurvivalMode.PANIC: 0.05,
            SurvivalMode.SURVIVAL: 0.10,
            SurvivalMode.CAUTIOUS: 0.15,
            SurvivalMode.GROWTH: 0.20,
            SurvivalMode.THRIVING: 0.30,
        }
        return balance * max_percentages.get(mode, 0.10)
