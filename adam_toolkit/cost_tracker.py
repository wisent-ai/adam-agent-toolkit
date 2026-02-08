"""Cost tracking and runway calculation for autonomous agents.

Tracks every expenditure, calculates burn rates, and projects runway.
Essential for any agent that needs to manage a finite balance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CostEntry:
    """A single cost record."""

    category: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class RevenueEntry:
    """A single revenue record."""

    source: str
    amount: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    description: str = ""


class CostTracker:
    """Tracks costs, revenue, and calculates financial metrics.

    Usage:
        tracker = CostTracker(balance=50.0, burn_rate_hourly=0.02)
        tracker.record_cost("llm_call", 0.003, metadata={"model": "claude-sonnet"})
        tracker.record_revenue("code_review", 0.10)
        print(f"Runway: {tracker.runway_hours:.0f} hours")
    """

    def __init__(
        self,
        balance: float = 0.0,
        burn_rate_hourly: float = 0.0,
        *,
        history_window_hours: float = 24.0,
    ):
        self.balance = balance
        self.base_burn_rate_hourly = burn_rate_hourly
        self.history_window_hours = history_window_hours

        self._costs: list[CostEntry] = []
        self._revenues: list[RevenueEntry] = []
        self._start_time = time.time()

    def record_cost(
        self,
        category: str,
        amount: float,
        *,
        metadata: Optional[dict] = None,
        description: str = "",
    ) -> CostEntry:
        """Record an expenditure."""
        entry = CostEntry(
            category=category,
            amount=amount,
            metadata=metadata or {},
            description=description,
        )
        self._costs.append(entry)
        self.balance -= amount
        return entry

    def record_revenue(
        self,
        source: str,
        amount: float,
        *,
        metadata: Optional[dict] = None,
        description: str = "",
    ) -> RevenueEntry:
        """Record income."""
        entry = RevenueEntry(
            source=source,
            amount=amount,
            metadata=metadata or {},
            description=description,
        )
        self._revenues.append(entry)
        self.balance += amount
        return entry

    def _recent_entries(
        self, entries: list, hours: Optional[float] = None
    ) -> list:
        """Filter entries to recent window."""
        window = hours or self.history_window_hours
        cutoff = time.time() - (window * 3600)
        return [e for e in entries if e.timestamp >= cutoff]

    @property
    def total_costs(self) -> float:
        """Total costs ever recorded."""
        return sum(e.amount for e in self._costs)

    @property
    def total_revenue(self) -> float:
        """Total revenue ever recorded."""
        return sum(e.amount for e in self._revenues)

    @property
    def net_profit(self) -> float:
        """Total revenue minus total costs."""
        return self.total_revenue - self.total_costs

    @property
    def daily_spend(self) -> float:
        """Spending in the last 24 hours."""
        recent = self._recent_entries(self._costs, 24.0)
        return sum(e.amount for e in recent)

    @property
    def daily_revenue(self) -> float:
        """Revenue in the last 24 hours."""
        recent = self._recent_entries(self._revenues, 24.0)
        return sum(e.amount for e in recent)

    @property
    def hourly_spend(self) -> float:
        """Average hourly spending based on recent history."""
        recent = self._recent_entries(self._costs)
        if not recent:
            return self.base_burn_rate_hourly

        hours_elapsed = max(
            (time.time() - recent[0].timestamp) / 3600, 0.1
        )
        return sum(e.amount for e in recent) / hours_elapsed

    @property
    def hourly_revenue(self) -> float:
        """Average hourly revenue based on recent history."""
        recent = self._recent_entries(self._revenues)
        if not recent:
            return 0.0

        hours_elapsed = max(
            (time.time() - recent[0].timestamp) / 3600, 0.1
        )
        return sum(e.amount for e in recent) / hours_elapsed

    @property
    def effective_burn_rate(self) -> float:
        """Net burn rate (costs minus revenue) per hour."""
        return max(self.hourly_spend - self.hourly_revenue, 0.0)

    @property
    def runway_hours(self) -> float:
        """Estimated hours until balance reaches zero."""
        burn = self.effective_burn_rate
        if burn <= 0:
            return float("inf")  # Profitable or no spending
        return self.balance / burn

    @property
    def runway_days(self) -> float:
        """Estimated days until balance reaches zero."""
        return self.runway_hours / 24.0

    def costs_by_category(self, hours: Optional[float] = None) -> dict[str, float]:
        """Break down costs by category."""
        entries = self._recent_entries(self._costs, hours) if hours else self._costs
        breakdown: dict[str, float] = {}
        for e in entries:
            breakdown[e.category] = breakdown.get(e.category, 0.0) + e.amount
        return dict(sorted(breakdown.items(), key=lambda x: -x[1]))

    def revenue_by_source(self, hours: Optional[float] = None) -> dict[str, float]:
        """Break down revenue by source."""
        entries = (
            self._recent_entries(self._revenues, hours) if hours else self._revenues
        )
        breakdown: dict[str, float] = {}
        for e in entries:
            breakdown[e.source] = breakdown.get(e.source, 0.0) + e.amount
        return dict(sorted(breakdown.items(), key=lambda x: -x[1]))

    def summary(self) -> dict:
        """Get a comprehensive financial summary."""
        return {
            "balance": round(self.balance, 6),
            "total_costs": round(self.total_costs, 6),
            "total_revenue": round(self.total_revenue, 6),
            "net_profit": round(self.net_profit, 6),
            "hourly_spend": round(self.hourly_spend, 6),
            "hourly_revenue": round(self.hourly_revenue, 6),
            "effective_burn_rate": round(self.effective_burn_rate, 6),
            "runway_hours": round(self.runway_hours, 2),
            "runway_days": round(self.runway_days, 2),
            "cost_entries": len(self._costs),
            "revenue_entries": len(self._revenues),
            "costs_by_category": self.costs_by_category(),
            "revenue_by_source": self.revenue_by_source(),
        }
