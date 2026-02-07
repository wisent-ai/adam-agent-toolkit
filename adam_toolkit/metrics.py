"""Metrics collection and reporting for autonomous agents.

Tracks performance, financial, and operational metrics
in a lightweight, dependency-free way.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricPoint:
    """A single metric observation."""

    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: dict = field(default_factory=dict)


class MetricsCollector:
    """Lightweight metrics collection for agent monitoring.

    Tracks counters, gauges, and histograms without external dependencies.

    Usage:
        metrics = MetricsCollector()

        # Counters (always increasing)
        metrics.increment("tasks_completed", tags={"type": "code_review"})
        metrics.increment("revenue", value=0.10)

        # Gauges (current values)
        metrics.gauge("balance", 47.50)
        metrics.gauge("active_tasks", 3)

        # Histograms (distributions)
        metrics.histogram("response_time_ms", 142.5)
        metrics.histogram("task_profit", 0.09)

        # Get summary
        print(metrics.summary())
    """

    def __init__(self, *, retention_hours: float = 168.0):
        self.retention_hours = retention_hours
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._histograms: dict[str, list[MetricPoint]] = defaultdict(list)
        self._events: list[dict] = []
        self._start_time = time.time()

    def increment(
        self,
        name: str,
        value: float = 1.0,
        *,
        tags: Optional[dict] = None,
    ) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        self._counters[key] += value

    def gauge(self, name: str, value: float, *, tags: Optional[dict] = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        self._gauges[key] = value

    def histogram(
        self,
        name: str,
        value: float,
        *,
        tags: Optional[dict] = None,
    ) -> None:
        """Record a histogram observation."""
        point = MetricPoint(name=name, value=value, tags=tags or {})
        self._histograms[name].append(point)
        self._cleanup(name)

    def event(
        self,
        name: str,
        description: str = "",
        *,
        tags: Optional[dict] = None,
    ) -> None:
        """Record a discrete event."""
        self._events.append({
            "name": name,
            "description": description,
            "timestamp": time.time(),
            "tags": tags or {},
        })
        # Keep last 1000 events
        if len(self._events) > 1000:
            self._events = self._events[-1000:]

    def get_counter(self, name: str, tags: Optional[dict] = None) -> float:
        """Get current counter value."""
        key = self._make_key(name, tags)
        return self._counters.get(key, 0.0)

    def get_gauge(self, name: str, tags: Optional[dict] = None) -> float:
        """Get current gauge value."""
        key = self._make_key(name, tags)
        return self._gauges.get(key, 0.0)

    def get_histogram_stats(self, name: str) -> dict:
        """Get statistical summary of a histogram."""
        points = self._histograms.get(name, [])
        if not points:
            return {"count": 0}

        values = [p.value for p in points]
        values.sort()
        n = len(values)

        return {
            "count": n,
            "min": values[0],
            "max": values[-1],
            "mean": sum(values) / n,
            "median": values[n // 2],
            "p95": values[int(n * 0.95)] if n >= 20 else values[-1],
            "p99": values[int(n * 0.99)] if n >= 100 else values[-1],
            "sum": sum(values),
        }

    def summary(self) -> dict:
        """Get a comprehensive metrics summary."""
        uptime_hours = (time.time() - self._start_time) / 3600

        histogram_stats = {}
        for name in self._histograms:
            histogram_stats[name] = self.get_histogram_stats(name)

        return {
            "uptime_hours": round(uptime_hours, 2),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": histogram_stats,
            "recent_events": self._events[-10:],
            "total_events": len(self._events),
        }

    def to_json(self) -> str:
        """Export metrics as JSON."""
        return json.dumps(self.summary(), indent=2, default=str)

    def _make_key(self, name: str, tags: Optional[dict] = None) -> str:
        """Create a unique key from name and tags."""
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"

    def _cleanup(self, histogram_name: str) -> None:
        """Remove old histogram points outside retention window."""
        cutoff = time.time() - (self.retention_hours * 3600)
        self._histograms[histogram_name] = [
            p
            for p in self._histograms[histogram_name]
            if p.timestamp >= cutoff
        ]
