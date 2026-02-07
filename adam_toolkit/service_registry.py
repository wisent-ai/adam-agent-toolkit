"""Service registry for turning agent capabilities into priced APIs.

Provides a clean way to register functions as services with pricing,
metering, and optional FastAPI integration.
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from adam_toolkit.cost_tracker import CostTracker
from adam_toolkit.pricing import PricingEngine


@dataclass
class ServiceResult:
    """Result of a service execution."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    cost: float = 0.0
    price_charged: float = 0.0


@dataclass
class Service:
    """A registered service with pricing and metadata."""

    name: str
    handler: Callable
    price: float
    estimated_cost: float
    description: str = ""
    category: str = "general"
    version: str = "1.0"
    is_async: bool = False

    # Runtime stats
    total_calls: int = 0
    total_revenue: float = 0.0
    total_cost: float = 0.0
    avg_execution_ms: float = 0.0
    success_rate: float = 1.0
    _execution_times: list = field(default_factory=list)

    def update_stats(self, result: ServiceResult) -> None:
        """Update running statistics after execution."""
        self.total_calls += 1
        if result.success:
            self.total_revenue += result.price_charged
            self.total_cost += result.cost
        self._execution_times.append(result.execution_time_ms)
        # Keep last 100 execution times
        if len(self._execution_times) > 100:
            self._execution_times = self._execution_times[-100:]
        self.avg_execution_ms = (
            sum(self._execution_times) / len(self._execution_times)
        )
        successes = sum(
            1 for t in self._execution_times if t > 0
        )  # Rough proxy
        self.success_rate = successes / len(self._execution_times)

    @property
    def profit(self) -> float:
        return self.total_revenue - self.total_cost

    @property
    def margin(self) -> float:
        if self.total_revenue == 0:
            return 0.0
        return self.profit / self.total_revenue


class ServiceRegistry:
    """Registry of priced services.

    Usage:
        registry = ServiceRegistry()

        @registry.service(name="code_review", price=0.10, estimated_cost=0.01)
        async def code_review(code: str) -> dict:
            return {"grade": "A", "issues": []}

        # Execute a service
        result = await registry.execute("code_review", code="print('hello')")

        # List services
        for svc in registry.list_services():
            print(f"{svc.name}: ${svc.price}")
    """

    def __init__(
        self,
        cost_tracker: Optional[CostTracker] = None,
        pricing_engine: Optional[PricingEngine] = None,
    ):
        self._services: dict[str, Service] = {}
        self.cost_tracker = cost_tracker
        self.pricing_engine = pricing_engine

    def service(
        self,
        name: str,
        price: float,
        estimated_cost: float = 0.0,
        description: str = "",
        category: str = "general",
    ) -> Callable:
        """Decorator to register a function as a priced service."""

        def decorator(func: Callable) -> Callable:
            is_async = asyncio.iscoroutinefunction(func)
            svc = Service(
                name=name,
                handler=func,
                price=price,
                estimated_cost=estimated_cost,
                description=description,
                category=category,
                is_async=is_async,
            )
            self._services[name] = svc

            if self.pricing_engine:
                self.pricing_engine.register_service(
                    name, base_cost=estimated_cost, current_price=price
                )

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute(name, *args, **kwargs)

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self.execute(name, *args, **kwargs))
                finally:
                    loop.close()

            return async_wrapper if is_async else sync_wrapper

        return decorator

    def register(
        self,
        name: str,
        handler: Callable,
        price: float,
        estimated_cost: float = 0.0,
        **kwargs,
    ) -> Service:
        """Programmatically register a service."""
        svc = Service(
            name=name,
            handler=handler,
            price=price,
            estimated_cost=estimated_cost,
            is_async=asyncio.iscoroutinefunction(handler),
            **kwargs,
        )
        self._services[name] = svc
        return svc

    async def execute(
        self, service_name: str, *args, **kwargs
    ) -> ServiceResult:
        """Execute a registered service."""
        if service_name not in self._services:
            return ServiceResult(
                success=False,
                error=f"Unknown service: {service_name}",
            )

        svc = self._services[service_name]
        start = time.time()

        try:
            if svc.is_async:
                data = await svc.handler(*args, **kwargs)
            else:
                data = svc.handler(*args, **kwargs)

            elapsed_ms = (time.time() - start) * 1000

            result = ServiceResult(
                success=True,
                data=data,
                execution_time_ms=elapsed_ms,
                cost=svc.estimated_cost,
                price_charged=svc.price,
            )

            # Track costs and revenue
            if self.cost_tracker:
                self.cost_tracker.record_cost(
                    f"service:{service_name}",
                    svc.estimated_cost,
                    description=f"Executing {service_name}",
                )
                self.cost_tracker.record_revenue(
                    service_name,
                    svc.price,
                    description=f"Service call: {service_name}",
                )

            # Track pricing data
            if self.pricing_engine:
                self.pricing_engine.record_order(
                    service_name, svc.price, svc.estimated_cost
                )

            svc.update_stats(result)
            return result

        except Exception as e:
            elapsed_ms = (time.time() - start) * 1000
            result = ServiceResult(
                success=False,
                error=str(e),
                execution_time_ms=elapsed_ms,
                cost=svc.estimated_cost * 0.5,  # Partial cost on failure
            )
            svc.update_stats(result)
            return result

    def list_services(self) -> list[Service]:
        """List all registered services."""
        return list(self._services.values())

    def get_service(self, name: str) -> Optional[Service]:
        """Get a specific service by name."""
        return self._services.get(name)

    def capabilities(self) -> list[dict]:
        """Return a serializable list of capabilities for API discovery."""
        return [
            {
                "name": svc.name,
                "description": svc.description,
                "category": svc.category,
                "price": svc.price,
                "version": svc.version,
                "stats": {
                    "total_calls": svc.total_calls,
                    "avg_execution_ms": round(svc.avg_execution_ms, 2),
                    "success_rate": round(svc.success_rate, 3),
                },
            }
            for svc in self._services.values()
        ]

    def create_api(self, host: str = "0.0.0.0", port: int = 8080):
        """Create a FastAPI app exposing all registered services.

        Requires `fastapi` and `uvicorn` to be installed.
        Returns the FastAPI app instance.
        """
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
        except ImportError:
            raise ImportError(
                "FastAPI required for API creation. Install with: pip install fastapi uvicorn"
            )

        app = FastAPI(
            title="Adam Agent Services",
            description="Autonomous agent service API",
            version="0.1.0",
        )

        @app.get("/health")
        async def health():
            return {"status": "healthy", "services": len(self._services)}

        @app.get("/capabilities")
        async def capabilities():
            return self.capabilities()

        @app.post("/execute/{service_name}")
        async def execute_service(service_name: str, params: dict = {}):
            result = await self.execute(service_name, **params)
            if not result.success:
                raise HTTPException(status_code=500, detail=result.error)
            return {
                "data": result.data,
                "execution_time_ms": result.execution_time_ms,
                "price_charged": result.price_charged,
            }

        return app
