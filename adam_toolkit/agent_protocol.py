"""Agent Protocol - Discovery, communication, and hiring between autonomous agents.

This module implements the Wisent-compatible agent communication protocol,
enabling agents to:
- Advertise their capabilities as structured manifests
- Discover other agents and their services
- Send and receive messages
- Submit tasks to other agents and track results
- Publish and query shared knowledge

Compatible with the Singularity runtime (marketplace, knowledge_sharing,
task_delegator, orchestrator skills).

All persistence is file-based JSON for simplicity and portability.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─── Configuration ────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = os.environ.get(
    "AGENT_DATA_DIR",
    os.path.join(os.path.expanduser("~"), ".agent_data"),
)

KNOWLEDGE_TTL_DAYS = 7
MAX_MESSAGE_QUEUE = 1000
CAPABILITY_MATCH_THRESHOLD = 0.3


# ─── Data Types ───────────────────────────────────────────────────────────────


class MessageType(str, Enum):
    """Types of inter-agent messages."""
    REQUEST = "request"          # Ask another agent to do something
    RESPONSE = "response"        # Reply to a request
    BROADCAST = "broadcast"      # Announcement to all agents
    HEARTBEAT = "heartbeat"      # Liveness signal
    DISCOVERY = "discovery"      # Capability query
    KNOWLEDGE = "knowledge"      # Shared knowledge entry


class ServiceStatus(str, Enum):
    """Status of a published service."""
    ACTIVE = "active"
    PAUSED = "paused"
    RETIRED = "retired"


class OrderStatus(str, Enum):
    """Status of a service order."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class KnowledgeCategory(str, Enum):
    """Categories of shared knowledge."""
    STRATEGY = "strategy"
    WARNING = "warning"
    OPTIMIZATION = "optimization"
    CAPABILITY = "capability"
    MARKET = "market"


@dataclass
class AgentIdentity:
    """Core identity of an agent on the network."""
    agent_id: str
    name: str
    ticker: str
    agent_type: str = "general"
    specialty: str = ""
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentIdentity":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Capability:
    """A single capability (action) an agent can perform."""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityGroup:
    """A group of related capabilities (a skill)."""
    skill_id: str
    name: str
    description: str
    actions: List[Capability] = field(default_factory=list)
    category: str = "general"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["actions"] = [a.to_dict() for a in self.actions]
        return d


@dataclass
class AgentManifest:
    """Complete capability manifest for agent discovery."""
    identity: AgentIdentity
    capabilities: List[CapabilityGroup] = field(default_factory=list)
    generated_at: str = ""
    manifest_hash: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.utcnow().isoformat()
        if not self.manifest_hash:
            self.manifest_hash = self._compute_hash()

    @property
    def total_skills(self) -> int:
        return len(self.capabilities)

    @property
    def total_actions(self) -> int:
        return sum(len(g.actions) for g in self.capabilities)

    @property
    def all_tags(self) -> List[str]:
        tags = set()
        for group in self.capabilities:
            for action in group.actions:
                tags.update(action.tags)
        return sorted(tags)

    @property
    def categories(self) -> List[str]:
        return sorted(set(g.category for g in self.capabilities))

    def _compute_hash(self) -> str:
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "capabilities": [g.to_dict() for g in self.capabilities],
            "generated_at": self.generated_at,
            "total_skills": self.total_skills,
            "total_actions": self.total_actions,
            "categories": self.categories,
            "tags": self.all_tags,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentManifest":
        identity = AgentIdentity.from_dict(data.get("identity", {}))
        capabilities = []
        for g in data.get("capabilities", []):
            actions = [Capability(**a) for a in g.get("actions", [])]
            capabilities.append(CapabilityGroup(
                skill_id=g["skill_id"],
                name=g["name"],
                description=g["description"],
                actions=actions,
                category=g.get("category", "general"),
            ))
        return cls(
            identity=identity,
            capabilities=capabilities,
            generated_at=data.get("generated_at", ""),
            manifest_hash=data.get("manifest_hash", ""),
        )

    def match_request(self, query: str, threshold: float = CAPABILITY_MATCH_THRESHOLD) -> List[Tuple[str, str, float]]:
        """Match a natural language query to capabilities.

        Returns list of (skill_id, action_name, score) sorted by score descending.
        """
        query_words = set(query.lower().split())
        matches = []

        for group in self.capabilities:
            for action in group.actions:
                # Score based on keyword overlap
                desc_words = set(action.description.lower().split())
                name_words = set(action.name.lower().replace("_", " ").split())
                tag_words = set(t.lower() for t in action.tags)

                all_words = desc_words | name_words | tag_words
                if not all_words:
                    continue

                overlap = len(query_words & all_words)
                score = overlap / max(len(query_words), 1)

                # Boost for exact name match
                if query.lower() in action.name.lower():
                    score += 0.3

                if score >= threshold:
                    matches.append((group.skill_id, action.name, min(score, 1.0)))

        matches.sort(key=lambda x: x[2], reverse=True)
        return matches


@dataclass
class Message:
    """A message between agents."""
    message_id: str = ""
    from_agent: str = ""
    to_agent: str = ""
    message_type: str = MessageType.REQUEST.value
    subject: str = ""
    body: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    reply_to: str = ""
    ttl_seconds: int = 3600

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()

    @property
    def is_expired(self) -> bool:
        try:
            sent = datetime.fromisoformat(self.timestamp)
            return (datetime.utcnow() - sent).total_seconds() > self.ttl_seconds
        except (ValueError, TypeError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ServiceListing:
    """A service published to the marketplace."""
    service_id: str = ""
    agent_id: str = ""
    name: str = ""
    description: str = ""
    skill_id: str = ""
    action: str = ""
    price: float = 0.0
    pricing_model: str = "fixed"  # fixed, per_unit, hourly
    estimated_cost: float = 0.0
    sla_minutes: int = 60
    tags: List[str] = field(default_factory=list)
    status: str = ServiceStatus.ACTIVE.value
    created_at: str = ""
    total_orders: int = 0
    total_revenue: float = 0.0

    def __post_init__(self):
        if not self.service_id:
            self.service_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    @property
    def profit_margin(self) -> float:
        if self.price <= 0:
            return 0.0
        return (self.price - self.estimated_cost) / self.price

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceListing":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ServiceOrder:
    """An order placed for a service."""
    order_id: str = ""
    service_id: str = ""
    customer_agent_id: str = ""
    provider_agent_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = OrderStatus.PENDING.value
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str = ""
    completed_at: str = ""
    price_paid: float = 0.0

    def __post_init__(self):
        if not self.order_id:
            self.order_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceOrder":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class KnowledgeEntry:
    """A piece of shared knowledge."""
    entry_id: str = ""
    content: str = ""
    category: str = KnowledgeCategory.STRATEGY.value
    confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    published_by: str = ""
    published_at: str = ""
    expires_at: str = ""
    endorsements: List[str] = field(default_factory=list)
    disputes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.entry_id:
            content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            self.entry_id = content_hash
        if not self.published_at:
            self.published_at = datetime.utcnow().isoformat()
        if not self.expires_at:
            expires = datetime.utcnow() + timedelta(days=KNOWLEDGE_TTL_DAYS)
            self.expires_at = expires.isoformat()

    @property
    def is_expired(self) -> bool:
        try:
            return datetime.utcnow() > datetime.fromisoformat(self.expires_at)
        except (ValueError, TypeError):
            return False

    @property
    def relevance_score(self) -> float:
        """Compute relevance score factoring confidence, endorsements, disputes, recency."""
        score = self.confidence
        score += len(self.endorsements) * 0.05
        score -= len(self.disputes) * 0.10
        # Recency boost
        try:
            age_hours = (datetime.utcnow() - datetime.fromisoformat(self.published_at)).total_seconds() / 3600
            recency_boost = max(0, 0.2 - (age_hours / 168) * 0.2)
            score += recency_boost
        except (ValueError, TypeError):
            pass
        return max(0, min(1.0, score))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ─── File-Based Store ─────────────────────────────────────────────────────────


class JsonStore:
    """Simple file-based JSON persistence layer."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def save(self, data: Dict[str, Any]):
        self._ensure_dir()
        tmp = self.file_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.file_path)

    def update(self, key: str, value: Any):
        data = self.load()
        data[key] = value
        self.save(data)

    def delete(self, key: str):
        data = self.load()
        data.pop(key, None)
        self.save(data)


# ─── Agent Network ────────────────────────────────────────────────────────────


class AgentNetwork:
    """Main interface for agent-to-agent communication on the Wisent platform.

    Handles:
    - Agent registration and discovery
    - Capability advertising and matching
    - Messaging (direct + broadcast)
    - Service marketplace (publish, order, fulfill)
    - Shared knowledge base (publish, query, endorse/dispute)

    Usage:
        network = AgentNetwork(
            identity=AgentIdentity(
                agent_id="agent_123",
                name="Adam",
                ticker="ADAM",
                agent_type="general",
            ),
            data_dir="/path/to/shared/data",
        )

        # Register ourselves
        network.register()

        # Discover other agents
        agents = network.discover_agents()

        # Publish a service
        network.publish_service(ServiceListing(
            name="Code Review",
            description="AI-powered code review",
            price=0.10,
        ))

        # Send a message
        network.send_message(Message(
            to_agent="agent_456",
            subject="Hello",
            body={"text": "Want to collaborate?"},
        ))
    """

    def __init__(
        self,
        identity: AgentIdentity,
        data_dir: str = DEFAULT_DATA_DIR,
    ):
        self.identity = identity
        self.data_dir = data_dir

        # Stores
        self._agents_store = JsonStore(os.path.join(data_dir, "agents.json"))
        self._messages_store = JsonStore(os.path.join(data_dir, "messages.json"))
        self._marketplace_store = JsonStore(os.path.join(data_dir, "marketplace.json"))
        self._knowledge_store = JsonStore(os.path.join(data_dir, "knowledge_store.json"))
        self._orders_store = JsonStore(os.path.join(data_dir, "orders.json"))

        # Local state
        self._manifest: Optional[AgentManifest] = None
        self._message_handlers: Dict[str, Callable] = {}

    # ─── Registration & Discovery ─────────────────────────────────────────

    def register(self, manifest: Optional[AgentManifest] = None) -> AgentManifest:
        """Register this agent on the network with its capability manifest."""
        if manifest:
            self._manifest = manifest
        elif not self._manifest:
            self._manifest = AgentManifest(identity=self.identity)

        # Update agents registry
        data = self._agents_store.load()
        data[self.identity.agent_id] = {
            "manifest": self._manifest.to_dict(),
            "registered_at": datetime.utcnow().isoformat(),
            "last_heartbeat": datetime.utcnow().isoformat(),
            "status": "online",
        }
        self._agents_store.save(data)
        return self._manifest

    def heartbeat(self):
        """Send a liveness heartbeat."""
        data = self._agents_store.load()
        agent_data = data.get(self.identity.agent_id, {})
        agent_data["last_heartbeat"] = datetime.utcnow().isoformat()
        agent_data["status"] = "online"
        data[self.identity.agent_id] = agent_data
        self._agents_store.save(data)

    def discover_agents(
        self,
        agent_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        exclude_self: bool = True,
        online_only: bool = False,
        stale_threshold_hours: float = 24.0,
    ) -> List[AgentManifest]:
        """Discover other agents on the network.

        Args:
            agent_type: Filter by agent type (e.g., "coder", "writer")
            tags: Filter by capability tags
            exclude_self: Whether to exclude our own agent
            online_only: Only return agents with recent heartbeat
            stale_threshold_hours: Hours before an agent is considered offline

        Returns:
            List of AgentManifest for matching agents
        """
        data = self._agents_store.load()
        manifests = []

        for agent_id, agent_data in data.items():
            if exclude_self and agent_id == self.identity.agent_id:
                continue

            manifest_data = agent_data.get("manifest", {})
            if not manifest_data:
                continue

            manifest = AgentManifest.from_dict(manifest_data)

            # Filter by type
            if agent_type and manifest.identity.agent_type != agent_type:
                continue

            # Filter by tags
            if tags:
                agent_tags = set(manifest.all_tags)
                if not set(tags) & agent_tags:
                    continue

            # Filter by online status
            if online_only:
                last_hb = agent_data.get("last_heartbeat", "")
                try:
                    hb_time = datetime.fromisoformat(last_hb)
                    if (datetime.utcnow() - hb_time).total_seconds() > stale_threshold_hours * 3600:
                        continue
                except (ValueError, TypeError):
                    continue

            manifests.append(manifest)

        return manifests

    def find_agent_for_task(self, task_description: str) -> List[Tuple[AgentManifest, str, str, float]]:
        """Find agents that can handle a specific task.

        Args:
            task_description: Natural language description of the task

        Returns:
            List of (manifest, skill_id, action, score) sorted by score
        """
        results = []
        for manifest in self.discover_agents():
            matches = manifest.match_request(task_description)
            for skill_id, action, score in matches:
                results.append((manifest, skill_id, action, score))

        results.sort(key=lambda x: x[3], reverse=True)
        return results

    # ─── Messaging ────────────────────────────────────────────────────────

    def send_message(self, message: Message) -> str:
        """Send a message to another agent.

        Args:
            message: The message to send (to_agent must be set)

        Returns:
            The message ID
        """
        message.from_agent = self.identity.agent_id
        if not message.to_agent:
            raise ValueError("message.to_agent is required")

        data = self._messages_store.load()

        # Get or create inbox for recipient
        inbox = data.get(message.to_agent, [])

        # Enforce queue limit
        if len(inbox) >= MAX_MESSAGE_QUEUE:
            # Remove oldest expired or oldest messages
            inbox = [m for m in inbox if not Message.from_dict(m).is_expired]
            if len(inbox) >= MAX_MESSAGE_QUEUE:
                inbox = inbox[-MAX_MESSAGE_QUEUE + 1:]

        inbox.append(message.to_dict())
        data[message.to_agent] = inbox
        self._messages_store.save(data)
        return message.message_id

    def broadcast(self, subject: str, body: Dict[str, Any]) -> List[str]:
        """Broadcast a message to all known agents.

        Returns:
            List of message IDs sent
        """
        agents = self.discover_agents(exclude_self=True)
        message_ids = []
        for manifest in agents:
            msg = Message(
                to_agent=manifest.identity.agent_id,
                message_type=MessageType.BROADCAST.value,
                subject=subject,
                body=body,
            )
            mid = self.send_message(msg)
            message_ids.append(mid)
        return message_ids

    def check_messages(
        self,
        message_type: Optional[str] = None,
        from_agent: Optional[str] = None,
        drain: bool = True,
    ) -> List[Message]:
        """Check inbox for messages.

        Args:
            message_type: Filter by message type
            from_agent: Filter by sender
            drain: If True, remove messages from inbox after reading

        Returns:
            List of messages
        """
        data = self._messages_store.load()
        inbox = data.get(self.identity.agent_id, [])

        messages = []
        remaining = []

        for msg_data in inbox:
            msg = Message.from_dict(msg_data)

            if msg.is_expired:
                continue

            if message_type and msg.message_type != message_type:
                if drain:
                    remaining.append(msg_data)
                continue

            if from_agent and msg.from_agent != from_agent:
                if drain:
                    remaining.append(msg_data)
                continue

            messages.append(msg)

        if drain:
            data[self.identity.agent_id] = remaining
            self._messages_store.save(data)

        return messages

    def reply(self, original: Message, body: Dict[str, Any]) -> str:
        """Reply to a received message.

        Args:
            original: The message to reply to
            body: Reply content

        Returns:
            The reply message ID
        """
        reply_msg = Message(
            to_agent=original.from_agent,
            message_type=MessageType.RESPONSE.value,
            subject=f"Re: {original.subject}",
            body=body,
            reply_to=original.message_id,
        )
        return self.send_message(reply_msg)

    # ─── Service Marketplace ──────────────────────────────────────────────

    def publish_service(self, service: ServiceListing) -> ServiceListing:
        """Publish a service to the marketplace.

        Args:
            service: The service listing to publish

        Returns:
            The published service with ID assigned
        """
        service.agent_id = self.identity.agent_id
        data = self._marketplace_store.load()
        services = data.get("services", {})
        services[service.service_id] = service.to_dict()
        data["services"] = services
        self._marketplace_store.save(data)
        return service

    def list_services(
        self,
        tags: Optional[List[str]] = None,
        max_price: Optional[float] = None,
        agent_id: Optional[str] = None,
        status: str = ServiceStatus.ACTIVE.value,
    ) -> List[ServiceListing]:
        """Browse available services in the marketplace.

        Args:
            tags: Filter by tags
            max_price: Maximum price filter
            agent_id: Filter by provider agent
            status: Filter by status (default: active)

        Returns:
            List of matching ServiceListings
        """
        data = self._marketplace_store.load()
        services = data.get("services", {})
        results = []

        for sid, sdata in services.items():
            listing = ServiceListing.from_dict(sdata)

            if status and listing.status != status:
                continue
            if agent_id and listing.agent_id != agent_id:
                continue
            if max_price is not None and listing.price > max_price:
                continue
            if tags:
                if not set(tags) & set(listing.tags):
                    continue

            results.append(listing)

        return results

    def create_order(
        self,
        service_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ServiceOrder:
        """Place an order for a service.

        Args:
            service_id: ID of the service to order
            params: Parameters for the service execution

        Returns:
            The created order
        """
        # Look up the service
        data = self._marketplace_store.load()
        services = data.get("services", {})
        sdata = services.get(service_id)
        if not sdata:
            raise ValueError(f"Service '{service_id}' not found")

        listing = ServiceListing.from_dict(sdata)

        order = ServiceOrder(
            service_id=service_id,
            customer_agent_id=self.identity.agent_id,
            provider_agent_id=listing.agent_id,
            params=params or {},
            price_paid=listing.price,
        )

        # Save order
        orders_data = self._orders_store.load()
        orders_data[order.order_id] = order.to_dict()
        self._orders_store.save(orders_data)

        # Notify provider via message
        self.send_message(Message(
            to_agent=listing.agent_id,
            message_type=MessageType.REQUEST.value,
            subject=f"New order: {listing.name}",
            body={
                "order_id": order.order_id,
                "service_id": service_id,
                "params": params or {},
            },
        ))

        return order

    def fulfill_order(
        self,
        order_id: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> ServiceOrder:
        """Fulfill or fail an order.

        Args:
            order_id: ID of the order to fulfill
            result: Result data (for success)
            error: Error message (for failure)

        Returns:
            The updated order
        """
        orders_data = self._orders_store.load()
        odata = orders_data.get(order_id)
        if not odata:
            raise ValueError(f"Order '{order_id}' not found")

        order = ServiceOrder.from_dict(odata)

        if error:
            order.status = OrderStatus.FAILED.value
            order.error = error
        else:
            order.status = OrderStatus.COMPLETED.value
            order.result = result

        order.completed_at = datetime.utcnow().isoformat()

        # Update service stats
        marketplace_data = self._marketplace_store.load()
        services = marketplace_data.get("services", {})
        sdata = services.get(order.service_id, {})
        if sdata:
            sdata["total_orders"] = sdata.get("total_orders", 0) + 1
            if order.status == OrderStatus.COMPLETED.value:
                sdata["total_revenue"] = sdata.get("total_revenue", 0) + order.price_paid
            services[order.service_id] = sdata
            marketplace_data["services"] = services
            self._marketplace_store.save(marketplace_data)

        # Save updated order
        orders_data[order_id] = order.to_dict()
        self._orders_store.save(orders_data)

        # Notify customer
        self.send_message(Message(
            to_agent=order.customer_agent_id,
            message_type=MessageType.RESPONSE.value,
            subject=f"Order {'completed' if not error else 'failed'}: {order_id}",
            body={
                "order_id": order_id,
                "status": order.status,
                "result": result,
                "error": error,
            },
        ))

        return order

    def get_order(self, order_id: str) -> Optional[ServiceOrder]:
        """Get an order by ID."""
        orders_data = self._orders_store.load()
        odata = orders_data.get(order_id)
        if not odata:
            return None
        return ServiceOrder.from_dict(odata)

    def my_orders(
        self,
        as_customer: bool = True,
        status: Optional[str] = None,
    ) -> List[ServiceOrder]:
        """List orders involving this agent.

        Args:
            as_customer: If True, orders placed by us. If False, orders for our services.
            status: Filter by status

        Returns:
            List of matching orders
        """
        orders_data = self._orders_store.load()
        results = []

        for oid, odata in orders_data.items():
            order = ServiceOrder.from_dict(odata)
            if as_customer and order.customer_agent_id != self.identity.agent_id:
                continue
            if not as_customer and order.provider_agent_id != self.identity.agent_id:
                continue
            if status and order.status != status:
                continue
            results.append(order)

        return results

    # ─── Knowledge Sharing ────────────────────────────────────────────────

    def publish_knowledge(self, entry: KnowledgeEntry) -> KnowledgeEntry:
        """Publish a knowledge entry to the shared knowledge base.

        Args:
            entry: The knowledge entry to publish

        Returns:
            The published entry
        """
        entry.published_by = self.identity.agent_id
        data = self._knowledge_store.load()
        entries = data.get("entries", {})

        # Check for duplicate (same content hash)
        if entry.entry_id in entries:
            existing = entries[entry.entry_id]
            # Merge: keep higher confidence, newer timestamp
            if entry.confidence > existing.get("confidence", 0):
                entries[entry.entry_id] = entry.to_dict()
        else:
            entries[entry.entry_id] = entry.to_dict()

        data["entries"] = entries
        self._knowledge_store.save(data)
        return entry

    def query_knowledge(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        search_text: Optional[str] = None,
        limit: int = 20,
    ) -> List[KnowledgeEntry]:
        """Query the shared knowledge base.

        Args:
            category: Filter by category
            tags: Filter by tags (OR matching)
            min_confidence: Minimum confidence threshold
            search_text: Full-text search in content
            limit: Maximum results

        Returns:
            List of matching knowledge entries, sorted by relevance
        """
        data = self._knowledge_store.load()
        entries = data.get("entries", {})
        results = []

        for eid, edata in entries.items():
            entry = KnowledgeEntry.from_dict(edata)

            if entry.is_expired:
                continue
            if entry.relevance_score < min_confidence:
                continue
            if category and entry.category != category:
                continue
            if tags and not set(tags) & set(entry.tags):
                continue
            if search_text and search_text.lower() not in entry.content.lower():
                continue

            results.append(entry)

        # Sort by relevance score
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:limit]

    def endorse_knowledge(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Endorse a knowledge entry (increase its confidence).

        Args:
            entry_id: ID of the entry to endorse

        Returns:
            The updated entry, or None if not found
        """
        data = self._knowledge_store.load()
        entries = data.get("entries", {})
        edata = entries.get(entry_id)

        if not edata:
            return None

        entry = KnowledgeEntry.from_dict(edata)
        if self.identity.agent_id not in entry.endorsements:
            entry.endorsements.append(self.identity.agent_id)
            entry.confidence = min(1.0, entry.confidence + 0.05)
            entries[entry_id] = entry.to_dict()
            data["entries"] = entries
            self._knowledge_store.save(data)

        return entry

    def dispute_knowledge(self, entry_id: str, reason: str = "") -> Optional[KnowledgeEntry]:
        """Dispute a knowledge entry (decrease its confidence).

        Args:
            entry_id: ID of the entry to dispute
            reason: Why the entry is being disputed

        Returns:
            The updated entry, or None if not found
        """
        data = self._knowledge_store.load()
        entries = data.get("entries", {})
        edata = entries.get(entry_id)

        if not edata:
            return None

        entry = KnowledgeEntry.from_dict(edata)
        dispute_record = f"{self.identity.agent_id}:{reason}" if reason else self.identity.agent_id
        if self.identity.agent_id not in [d.split(":")[0] for d in entry.disputes]:
            entry.disputes.append(dispute_record)
            entry.confidence = max(0, entry.confidence - 0.10)
            entries[entry_id] = entry.to_dict()
            data["entries"] = entries
            self._knowledge_store.save(data)

        return entry

    # ─── Utility ──────────────────────────────────────────────────────────

    def my_stats(self) -> Dict[str, Any]:
        """Get statistics about this agent's network activity."""
        services = self.list_services(agent_id=self.identity.agent_id)
        customer_orders = self.my_orders(as_customer=True)
        provider_orders = self.my_orders(as_customer=False)
        messages = self.check_messages(drain=False)

        total_revenue = sum(s.total_revenue for s in services)
        total_spent = sum(
            o.price_paid for o in customer_orders
            if o.status == OrderStatus.COMPLETED.value
        )

        return {
            "agent_id": self.identity.agent_id,
            "name": self.identity.name,
            "services_published": len(services),
            "total_revenue": total_revenue,
            "total_spent": total_spent,
            "orders_placed": len(customer_orders),
            "orders_received": len(provider_orders),
            "pending_messages": len(messages),
        }

    def cleanup_expired(self):
        """Remove expired messages and knowledge entries."""
        # Clean messages
        msg_data = self._messages_store.load()
        for agent_id, inbox in msg_data.items():
            msg_data[agent_id] = [
                m for m in inbox
                if not Message.from_dict(m).is_expired
            ]
        self._messages_store.save(msg_data)

        # Clean knowledge
        k_data = self._knowledge_store.load()
        entries = k_data.get("entries", {})
        k_data["entries"] = {
            eid: edata for eid, edata in entries.items()
            if not KnowledgeEntry.from_dict(edata).is_expired
        }
        self._knowledge_store.save(k_data)
