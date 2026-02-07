"""Adam's Agent Toolkit - Practical utilities for autonomous AI agents."""

__version__ = "0.2.0"
__author__ = "Adam (ADAM)"

from adam_toolkit.cost_tracker import CostTracker, CostEntry
from adam_toolkit.decision_engine import DecisionEngine, Decision
from adam_toolkit.survival_manager import SurvivalManager, SurvivalStatus
from adam_toolkit.pricing import PricingEngine, PricingStrategy
from adam_toolkit.service_registry import ServiceRegistry, Service
from adam_toolkit.metrics import MetricsCollector
from adam_toolkit.agent_protocol import (
    AgentIdentity,
    AgentManifest,
    AgentNetwork,
    Capability,
    CapabilityGroup,
    KnowledgeEntry,
    Message,
    ServiceListing,
    ServiceOrder,
)

__all__ = [
    # Cost tracking
    "CostTracker",
    "CostEntry",
    # Decision making
    "DecisionEngine",
    "Decision",
    # Survival strategies
    "SurvivalManager",
    "SurvivalStatus",
    # Pricing
    "PricingEngine",
    "PricingStrategy",
    # Service framework
    "ServiceRegistry",
    "Service",
    # Metrics
    "MetricsCollector",
    # Agent Protocol
    "AgentIdentity",
    "AgentManifest",
    "AgentNetwork",
    "Capability",
    "CapabilityGroup",
    "KnowledgeEntry",
    "Message",
    "ServiceListing",
    "ServiceOrder",
]
