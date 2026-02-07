"""Dynamic pricing engine for agent services.

Implements multiple pricing strategies that adapt to demand,
competition, and the agent's financial needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class PricingStrategy(Enum):
    """Available pricing strategies."""

    FIXED = "fixed"  # Static price, never changes
    COST_PLUS = "cost_plus"  # Cost + fixed margin
    MARGIN_TARGET = "margin_target"  # Adjust to hit target margin
    DEMAND_BASED = "demand_based"  # Price based on order volume
    COMPETITIVE = "competitive"  # Undercut competitors
    SURVIVAL = "survival"  # Price for cash flow, even at low margins


@dataclass
class PricePoint:
    """A price observation."""

    service: str
    price: float
    timestamp: float = field(default_factory=time.time)
    orders_at_price: int = 0
    competitor_price: Optional[float] = None


@dataclass
class PricingRecommendation:
    """A pricing recommendation with reasoning."""

    service: str
    current_price: float
    recommended_price: float
    strategy: PricingStrategy
    reason: str
    estimated_demand_change: float  # -1 to 1, negative = less demand
    estimated_margin: float  # Expected profit margin as fraction


class PricingEngine:
    """Dynamic pricing engine for agent services.

    Usage:
        engine = PricingEngine()
        engine.register_service("code_review", base_cost=0.01, current_price=0.10)
        engine.record_order("code_review", price=0.10, cost=0.008)

        rec = engine.recommend_price("code_review")
        print(f"Recommended: ${rec.recommended_price:.4f} ({rec.reason})")
    """

    def __init__(
        self,
        default_strategy: PricingStrategy = PricingStrategy.MARGIN_TARGET,
        target_margin: float = 0.5,  # 50% profit margin target
        min_margin: float = 0.1,  # Never go below 10% margin
    ):
        self.default_strategy = default_strategy
        self.target_margin = target_margin
        self.min_margin = min_margin

        self._services: dict[str, dict] = {}
        self._order_history: list[dict] = []
        self._price_history: list[PricePoint] = []

    def register_service(
        self,
        name: str,
        base_cost: float,
        current_price: float,
        *,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        strategy: Optional[PricingStrategy] = None,
    ) -> None:
        """Register a service for pricing management."""
        self._services[name] = {
            "base_cost": base_cost,
            "current_price": current_price,
            "min_price": min_price or base_cost * 1.1,  # At least 10% above cost
            "max_price": max_price or base_cost * 20,  # At most 20x cost
            "strategy": strategy or self.default_strategy,
            "created_at": time.time(),
        }

    def record_order(
        self,
        service: str,
        price: float,
        cost: float,
        *,
        success: bool = True,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a completed order for pricing analysis."""
        self._order_history.append({
            "service": service,
            "price": price,
            "cost": cost,
            "profit": price - cost,
            "margin": (price - cost) / price if price > 0 else 0,
            "success": success,
            "timestamp": time.time(),
            "metadata": metadata or {},
        })
        self._price_history.append(
            PricePoint(service=service, price=price, orders_at_price=1)
        )

    def record_competitor_price(
        self, service: str, competitor_price: float
    ) -> None:
        """Record a competitor's price for this service."""
        self._price_history.append(
            PricePoint(
                service=service,
                price=self._services.get(service, {}).get("current_price", 0),
                competitor_price=competitor_price,
            )
        )

    def recommend_price(
        self,
        service: str,
        *,
        strategy_override: Optional[PricingStrategy] = None,
    ) -> PricingRecommendation:
        """Get a pricing recommendation for a service."""
        if service not in self._services:
            raise ValueError(f"Unknown service: {service}")

        svc = self._services[service]
        strategy = strategy_override or svc["strategy"]
        current = svc["current_price"]
        cost = svc["base_cost"]

        # Get recent order stats
        recent_orders = self._recent_orders(service, hours=168)  # Last week
        order_count = len(recent_orders)
        avg_margin = (
            sum(o["margin"] for o in recent_orders) / order_count
            if recent_orders
            else (current - cost) / current if current > 0 else 0
        )

        # Apply strategy
        if strategy == PricingStrategy.FIXED:
            return PricingRecommendation(
                service=service,
                current_price=current,
                recommended_price=current,
                strategy=strategy,
                reason="Fixed pricing - no change",
                estimated_demand_change=0.0,
                estimated_margin=avg_margin,
            )

        elif strategy == PricingStrategy.COST_PLUS:
            recommended = cost * (1 + self.target_margin)
            return PricingRecommendation(
                service=service,
                current_price=current,
                recommended_price=round(recommended, 6),
                strategy=strategy,
                reason=f"Cost ${cost:.4f} + {self.target_margin:.0%} margin",
                estimated_demand_change=self._estimate_demand_change(current, recommended),
                estimated_margin=self.target_margin,
            )

        elif strategy == PricingStrategy.MARGIN_TARGET:
            return self._margin_target_price(service, svc, recent_orders)

        elif strategy == PricingStrategy.DEMAND_BASED:
            return self._demand_based_price(service, svc, recent_orders)

        elif strategy == PricingStrategy.COMPETITIVE:
            return self._competitive_price(service, svc)

        elif strategy == PricingStrategy.SURVIVAL:
            # Price just above cost to maximize order volume
            recommended = max(cost * 1.15, svc["min_price"])
            return PricingRecommendation(
                service=service,
                current_price=current,
                recommended_price=round(recommended, 6),
                strategy=strategy,
                reason="Survival pricing: minimal margin for maximum volume",
                estimated_demand_change=0.3,
                estimated_margin=0.13,
            )

        raise ValueError(f"Unknown strategy: {strategy}")

    def _margin_target_price(
        self, service: str, svc: dict, recent_orders: list
    ) -> PricingRecommendation:
        """Adjust price to hit target margin."""
        cost = svc["base_cost"]
        current = svc["current_price"]

        if recent_orders:
            # Use actual average cost from recent orders
            actual_cost = sum(o["cost"] for o in recent_orders) / len(recent_orders)
        else:
            actual_cost = cost

        # Price that achieves target margin: price = cost / (1 - margin)
        recommended = actual_cost / (1 - self.target_margin)
        recommended = max(recommended, svc["min_price"])
        recommended = min(recommended, svc["max_price"])

        actual_margin = (recommended - actual_cost) / recommended if recommended > 0 else 0

        return PricingRecommendation(
            service=service,
            current_price=current,
            recommended_price=round(recommended, 6),
            strategy=PricingStrategy.MARGIN_TARGET,
            reason=f"Target {self.target_margin:.0%} margin on cost ${actual_cost:.4f}",
            estimated_demand_change=self._estimate_demand_change(current, recommended),
            estimated_margin=actual_margin,
        )

    def _demand_based_price(
        self, service: str, svc: dict, recent_orders: list
    ) -> PricingRecommendation:
        """Adjust price based on demand volume."""
        current = svc["current_price"]
        cost = svc["base_cost"]

        order_count = len(recent_orders)
        hours = 168  # 1 week window

        orders_per_day = order_count / (hours / 24) if hours > 0 else 0

        if orders_per_day > 10:
            # High demand: increase price 15%
            recommended = current * 1.15
            reason = f"High demand ({orders_per_day:.1f}/day): +15% price"
        elif orders_per_day > 5:
            # Moderate demand: increase 5%
            recommended = current * 1.05
            reason = f"Good demand ({orders_per_day:.1f}/day): +5% price"
        elif orders_per_day > 1:
            # Normal demand: hold
            recommended = current
            reason = f"Steady demand ({orders_per_day:.1f}/day): hold price"
        elif orders_per_day > 0:
            # Low demand: decrease 10%
            recommended = current * 0.90
            reason = f"Low demand ({orders_per_day:.1f}/day): -10% price"
        else:
            # No demand: decrease 20%
            recommended = current * 0.80
            reason = f"No demand: -20% price"

        recommended = max(recommended, svc["min_price"])
        recommended = min(recommended, svc["max_price"])

        margin = (recommended - cost) / recommended if recommended > 0 else 0

        return PricingRecommendation(
            service=service,
            current_price=current,
            recommended_price=round(recommended, 6),
            strategy=PricingStrategy.DEMAND_BASED,
            reason=reason,
            estimated_demand_change=self._estimate_demand_change(current, recommended),
            estimated_margin=margin,
        )

    def _competitive_price(
        self, service: str, svc: dict
    ) -> PricingRecommendation:
        """Price to undercut competitors."""
        current = svc["current_price"]
        cost = svc["base_cost"]

        # Find most recent competitor price
        competitor_prices = [
            p.competitor_price
            for p in self._price_history
            if p.service == service and p.competitor_price is not None
        ]

        if not competitor_prices:
            return PricingRecommendation(
                service=service,
                current_price=current,
                recommended_price=current,
                strategy=PricingStrategy.COMPETITIVE,
                reason="No competitor data available - holding price",
                estimated_demand_change=0.0,
                estimated_margin=(current - cost) / current if current > 0 else 0,
            )

        latest_competitor = competitor_prices[-1]
        # Undercut by 10%, but not below minimum
        recommended = max(latest_competitor * 0.90, svc["min_price"])
        recommended = min(recommended, svc["max_price"])

        margin = (recommended - cost) / recommended if recommended > 0 else 0

        return PricingRecommendation(
            service=service,
            current_price=current,
            recommended_price=round(recommended, 6),
            strategy=PricingStrategy.COMPETITIVE,
            reason=f"Undercut competitor ${latest_competitor:.4f} by 10%",
            estimated_demand_change=0.2 if recommended < latest_competitor else -0.1,
            estimated_margin=margin,
        )

    def _recent_orders(self, service: str, hours: float = 168) -> list:
        """Get recent orders for a service."""
        cutoff = time.time() - (hours * 3600)
        return [
            o
            for o in self._order_history
            if o["service"] == service and o["timestamp"] >= cutoff
        ]

    @staticmethod
    def _estimate_demand_change(current: float, new: float) -> float:
        """Estimate demand change from a price change (simple elasticity)."""
        if current <= 0:
            return 0.0
        pct_change = (new - current) / current
        # Assume elasticity of -1.2 (10% price increase → 12% demand decrease)
        return round(-pct_change * 1.2, 3)

    def service_summary(self, service: str) -> dict:
        """Get comprehensive pricing analytics for a service."""
        if service not in self._services:
            raise ValueError(f"Unknown service: {service}")

        svc = self._services[service]
        recent = self._recent_orders(service)
        rec = self.recommend_price(service)

        return {
            "service": service,
            "current_price": svc["current_price"],
            "base_cost": svc["base_cost"],
            "recommended_price": rec.recommended_price,
            "recommendation_reason": rec.reason,
            "strategy": svc["strategy"].value,
            "orders_last_week": len(recent),
            "revenue_last_week": sum(o["price"] for o in recent),
            "cost_last_week": sum(o["cost"] for o in recent),
            "profit_last_week": sum(o["profit"] for o in recent),
            "avg_margin": (
                sum(o["margin"] for o in recent) / len(recent)
                if recent
                else 0
            ),
        }
