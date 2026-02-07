"""Tests for the pricing engine module."""

from adam_toolkit.pricing import PricingEngine, PricingStrategy


def test_fixed_pricing():
    engine = PricingEngine()
    engine.register_service(
        "test", base_cost=0.01, current_price=0.10,
        strategy=PricingStrategy.FIXED
    )

    rec = engine.recommend_price("test")
    assert rec.recommended_price == 0.10
    assert rec.strategy == PricingStrategy.FIXED


def test_cost_plus_pricing():
    engine = PricingEngine(target_margin=0.5)
    engine.register_service(
        "test", base_cost=0.01, current_price=0.10,
        strategy=PricingStrategy.COST_PLUS
    )

    rec = engine.recommend_price("test")
    # Cost 0.01 + 50% margin = 0.015
    assert rec.recommended_price == 0.015


def test_margin_target_pricing():
    engine = PricingEngine(target_margin=0.5)
    engine.register_service(
        "test", base_cost=0.01, current_price=0.05,
        strategy=PricingStrategy.MARGIN_TARGET
    )

    rec = engine.recommend_price("test")
    # Price = cost / (1 - margin) = 0.01 / 0.5 = 0.02
    assert rec.recommended_price == 0.02
    assert rec.estimated_margin >= 0.49  # ~50%


def test_survival_pricing():
    engine = PricingEngine()
    engine.register_service(
        "test", base_cost=0.01, current_price=0.10,
        strategy=PricingStrategy.SURVIVAL
    )

    rec = engine.recommend_price("test")
    # Survival: just above cost
    assert rec.recommended_price < 0.10
    assert rec.recommended_price >= 0.01 * 1.1  # At least above min


def test_demand_based_no_orders():
    engine = PricingEngine()
    engine.register_service(
        "test", base_cost=0.01, current_price=0.10,
        strategy=PricingStrategy.DEMAND_BASED
    )

    rec = engine.recommend_price("test")
    # No demand: should decrease price
    assert rec.recommended_price < 0.10


def test_competitive_pricing():
    engine = PricingEngine()
    engine.register_service(
        "test", base_cost=0.01, current_price=0.10,
        strategy=PricingStrategy.COMPETITIVE
    )

    engine.record_competitor_price("test", 0.08)

    rec = engine.recommend_price("test")
    # Should undercut competitor by 10%
    assert rec.recommended_price < 0.08
    assert rec.recommended_price >= 0.01 * 1.1  # Above min


def test_order_recording():
    engine = PricingEngine()
    engine.register_service("test", base_cost=0.01, current_price=0.10)

    engine.record_order("test", price=0.10, cost=0.008)
    engine.record_order("test", price=0.10, cost=0.012)

    summary = engine.service_summary("test")
    assert summary["orders_last_week"] == 2
    assert summary["revenue_last_week"] == 0.20
    assert summary["profit_last_week"] == 0.20 - 0.008 - 0.012


def test_min_max_price_bounds():
    engine = PricingEngine(target_margin=0.99)  # Very high margin target
    engine.register_service(
        "test", base_cost=0.01, current_price=0.10,
        max_price=0.50,
        strategy=PricingStrategy.MARGIN_TARGET
    )

    rec = engine.recommend_price("test")
    assert rec.recommended_price <= 0.50
