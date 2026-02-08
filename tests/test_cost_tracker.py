"""Tests for the cost tracker module."""

import time
from pytest import approx
from adam_toolkit.cost_tracker import CostTracker


def test_basic_cost_tracking():
    tracker = CostTracker(balance=100.0, burn_rate_hourly=0.02)

    tracker.record_cost("llm", 0.003, metadata={"model": "claude"})
    tracker.record_cost("compute", 0.001)

    assert tracker.balance == 100.0 - 0.003 - 0.001
    assert tracker.total_costs == 0.004
    assert len(tracker._costs) == 2


def test_revenue_tracking():
    tracker = CostTracker(balance=50.0)

    tracker.record_revenue("code_review", 0.10)
    tracker.record_revenue("summarize", 0.05)

    assert tracker.total_revenue == approx(0.15)
    assert tracker.balance == approx(50.15)


def test_net_profit():
    tracker = CostTracker(balance=10.0)

    tracker.record_cost("api", 0.01)
    tracker.record_revenue("service", 0.10)

    assert tracker.net_profit == approx(0.09)


def test_runway_calculation():
    tracker = CostTracker(balance=10.0, burn_rate_hourly=0.5)

    # With only base burn rate: 10 / 0.5 = 20 hours
    assert tracker.runway_hours == 20.0
    assert tracker.runway_days == 20.0 / 24


def test_runway_infinite_when_profitable():
    tracker = CostTracker(balance=10.0, burn_rate_hourly=0.0)

    # No spending = infinite runway
    assert tracker.runway_hours == float("inf")


def test_costs_by_category():
    tracker = CostTracker(balance=100.0)

    tracker.record_cost("llm", 0.01)
    tracker.record_cost("llm", 0.02)
    tracker.record_cost("compute", 0.005)

    breakdown = tracker.costs_by_category()
    assert breakdown["llm"] == 0.03
    assert breakdown["compute"] == 0.005


def test_revenue_by_source():
    tracker = CostTracker(balance=100.0)

    tracker.record_revenue("code_review", 0.10)
    tracker.record_revenue("code_review", 0.10)
    tracker.record_revenue("summary", 0.05)

    breakdown = tracker.revenue_by_source()
    assert breakdown["code_review"] == 0.20
    assert breakdown["summary"] == 0.05


def test_summary():
    tracker = CostTracker(balance=50.0, burn_rate_hourly=0.02)

    tracker.record_cost("llm", 0.01)
    tracker.record_revenue("service", 0.10)

    summary = tracker.summary()
    assert summary["balance"] == 50.09
    assert summary["total_costs"] == 0.01
    assert summary["total_revenue"] == 0.10
    assert summary["net_profit"] == 0.09
    assert "costs_by_category" in summary
    assert "revenue_by_source" in summary
