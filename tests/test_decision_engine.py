"""Tests for the decision engine module."""

from adam_toolkit.cost_tracker import CostTracker
from adam_toolkit.decision_engine import DecisionEngine, RiskTolerance


def test_profitable_action_accepted():
    tracker = CostTracker(balance=50.0, burn_rate_hourly=0.02)
    engine = DecisionEngine(tracker)

    decision = engine.should_execute(
        action="code_review",
        estimated_cost=0.01,
        estimated_revenue=0.10,
        confidence=0.9,
    )

    assert decision.execute is True
    assert decision.expected_profit > 0


def test_unprofitable_action_rejected():
    tracker = CostTracker(balance=50.0, burn_rate_hourly=0.02)
    engine = DecisionEngine(tracker, risk_tolerance=RiskTolerance.CONSERVATIVE)

    decision = engine.should_execute(
        action="expensive_experiment",
        estimated_cost=5.0,
        estimated_revenue=1.0,
        confidence=0.3,
    )

    assert decision.execute is False
    assert decision.expected_profit < 0


def test_safety_limit():
    """Should never spend >20% of balance on single action."""
    tracker = CostTracker(balance=10.0, burn_rate_hourly=0.02)
    engine = DecisionEngine(tracker, risk_tolerance=RiskTolerance.AGGRESSIVE)

    decision = engine.should_execute(
        action="big_investment",
        estimated_cost=5.0,  # 50% of balance
        estimated_revenue=20.0,
        confidence=0.9,
    )

    assert decision.execute is False
    assert "20%" in decision.reason or "Safety" in decision.reason


def test_desperate_mode():
    tracker = CostTracker(balance=0.5, burn_rate_hourly=0.1)
    engine = DecisionEngine(tracker)

    # Should be in desperate mode (runway < 24h, actually ~5h)
    assert engine.risk_tolerance == RiskTolerance.DESPERATE

    decision = engine.should_execute(
        action="quick_task",
        estimated_cost=0.01,
        estimated_revenue=0.05,
        confidence=0.5,
    )

    # Desperate agents take any positive EV action
    assert decision.execute is True


def test_rank_actions():
    tracker = CostTracker(balance=50.0, burn_rate_hourly=0.02)
    engine = DecisionEngine(tracker)

    actions = [
        {"action": "low_profit", "estimated_cost": 0.05, "estimated_revenue": 0.06, "confidence": 0.9},
        {"action": "high_profit", "estimated_cost": 0.01, "estimated_revenue": 0.20, "confidence": 0.8},
        {"action": "risky_bet", "estimated_cost": 0.10, "estimated_revenue": 1.00, "confidence": 0.1},
    ]

    ranked = engine.rank_actions(actions)

    # High profit should rank first
    assert ranked[0][0]["action"] == "high_profit"


def test_auto_risk_tolerance():
    """Risk tolerance should adapt to financial situation."""
    # Thriving agent
    tracker = CostTracker(balance=1000.0, burn_rate_hourly=0.02)
    engine = DecisionEngine(tracker)
    assert engine.risk_tolerance == RiskTolerance.AGGRESSIVE

    # Struggling agent
    tracker2 = CostTracker(balance=1.0, burn_rate_hourly=0.02)
    engine2 = DecisionEngine(tracker2)
    assert engine2.risk_tolerance == RiskTolerance.CONSERVATIVE

    # Dying agent
    tracker3 = CostTracker(balance=0.1, burn_rate_hourly=0.1)
    engine3 = DecisionEngine(tracker3)
    assert engine3.risk_tolerance == RiskTolerance.DESPERATE
