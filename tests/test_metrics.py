"""Tests for the metrics collector module."""

from adam_toolkit.metrics import MetricsCollector


def test_counter_increment():
    m = MetricsCollector()
    m.increment("tasks")
    m.increment("tasks")
    m.increment("tasks", value=3)

    assert m.get_counter("tasks") == 5


def test_counter_with_tags():
    m = MetricsCollector()
    m.increment("tasks", tags={"type": "review"})
    m.increment("tasks", tags={"type": "summary"})
    m.increment("tasks", tags={"type": "review"})

    assert m.get_counter("tasks", tags={"type": "review"}) == 2
    assert m.get_counter("tasks", tags={"type": "summary"}) == 1


def test_gauge():
    m = MetricsCollector()
    m.gauge("balance", 50.0)
    assert m.get_gauge("balance") == 50.0

    m.gauge("balance", 47.5)
    assert m.get_gauge("balance") == 47.5


def test_histogram_stats():
    m = MetricsCollector()

    values = [10, 20, 30, 40, 50]
    for v in values:
        m.histogram("response_time", v)

    stats = m.get_histogram_stats("response_time")
    assert stats["count"] == 5
    assert stats["min"] == 10
    assert stats["max"] == 50
    assert stats["mean"] == 30
    assert stats["sum"] == 150


def test_event_recording():
    m = MetricsCollector()
    m.event("task_completed", "Finished code review", tags={"service": "review"})
    m.event("error", "API timeout")

    summary = m.summary()
    assert summary["total_events"] == 2
    assert len(summary["recent_events"]) == 2


def test_summary():
    m = MetricsCollector()
    m.increment("revenue", value=0.10)
    m.gauge("balance", 50.0)
    m.histogram("latency", 100)

    summary = m.summary()
    assert "counters" in summary
    assert "gauges" in summary
    assert "histograms" in summary
    assert summary["uptime_hours"] >= 0


def test_json_export():
    m = MetricsCollector()
    m.increment("test")
    m.gauge("val", 42)

    json_str = m.to_json()
    assert "test" in json_str
    assert "42" in json_str
