"""Tests for the survival manager module."""

from adam_toolkit.cost_tracker import CostTracker
from adam_toolkit.survival_manager import SurvivalManager, SurvivalMode


def test_thriving_mode():
    tracker = CostTracker(balance=500.0, burn_rate_hourly=0.02)
    manager = SurvivalManager(tracker)

    status = manager.assess()

    assert status.mode == SurvivalMode.THRIVING
    assert status.runway_days > 7
    assert status.risk_budget > 0
    assert any("invest" in a.lower() or "long-term" in a.lower() for a in status.recommended_actions)


def test_panic_mode():
    tracker = CostTracker(balance=0.5, burn_rate_hourly=0.1)
    manager = SurvivalManager(tracker)

    status = manager.assess()

    assert status.mode == SurvivalMode.PANIC
    assert status.runway_hours < 6
    assert status.risk_budget == 0.0  # No risk in panic
    assert any("urgent" in a.lower() for a in status.recommended_actions)


def test_survival_mode():
    tracker = CostTracker(balance=2.0, burn_rate_hourly=0.1)
    manager = SurvivalManager(tracker)

    status = manager.assess()

    assert status.mode == SurvivalMode.SURVIVAL
    assert any("proven" in a.lower() or "minimize" in a.lower() for a in status.recommended_actions)


def test_is_profitable_flag():
    tracker = CostTracker(balance=50.0, burn_rate_hourly=0.02)

    # Record lots of revenue
    for _ in range(10):
        tracker.record_revenue("services", 0.10)

    manager = SurvivalManager(tracker)
    status = manager.assess()

    # Revenue rate should be positive, but is_profitable depends on comparison with burn
    assert status.revenue_rate_hourly > 0


def test_max_expense_scales_with_mode():
    # Thriving agent can spend more per action
    tracker1 = CostTracker(balance=100.0, burn_rate_hourly=0.001)
    status1 = SurvivalManager(tracker1).assess()

    # Panicking agent is very constrained
    tracker2 = CostTracker(balance=0.3, burn_rate_hourly=0.1)
    status2 = SurvivalManager(tracker2).assess()

    # Thriving can spend 30%, panic only 5%
    assert status1.max_single_expense > status2.max_single_expense
    assert status1.max_single_expense == 100.0 * 0.30
    assert status2.max_single_expense == 0.3 * 0.05
