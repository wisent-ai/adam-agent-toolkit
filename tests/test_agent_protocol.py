"""Tests for the agent_protocol module."""

import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta

from adam_toolkit.agent_protocol import (
    AgentIdentity,
    AgentManifest,
    AgentNetwork,
    Capability,
    CapabilityGroup,
    KnowledgeCategory,
    KnowledgeEntry,
    Message,
    MessageType,
    OrderStatus,
    ServiceListing,
    ServiceOrder,
    ServiceStatus,
)


class TestAgentIdentity(unittest.TestCase):
    def test_create_and_serialize(self):
        identity = AgentIdentity(
            agent_id="agent_123",
            name="TestAgent",
            ticker="TEST",
            agent_type="coder",
        )
        d = identity.to_dict()
        self.assertEqual(d["agent_id"], "agent_123")
        self.assertEqual(d["name"], "TestAgent")
        self.assertEqual(d["ticker"], "TEST")

    def test_from_dict(self):
        data = {"agent_id": "a1", "name": "Bob", "ticker": "BOB", "agent_type": "writer"}
        identity = AgentIdentity.from_dict(data)
        self.assertEqual(identity.name, "Bob")
        self.assertEqual(identity.agent_type, "writer")


class TestCapabilityMatching(unittest.TestCase):
    def setUp(self):
        self.manifest = AgentManifest(
            identity=AgentIdentity("a1", "TestAgent", "TEST"),
            capabilities=[
                CapabilityGroup(
                    skill_id="code",
                    name="Code Skills",
                    description="Code analysis and review",
                    actions=[
                        Capability(
                            name="code_review",
                            description="Review code for bugs and security issues",
                            tags=["code", "review", "security"],
                        ),
                        Capability(
                            name="summarize",
                            description="Summarize text into concise form",
                            tags=["text", "summary", "nlp"],
                        ),
                    ],
                ),
            ],
        )

    def test_match_by_keyword(self):
        matches = self.manifest.match_request("code review")
        self.assertTrue(len(matches) > 0)
        self.assertEqual(matches[0][1], "code_review")

    def test_match_by_tag(self):
        matches = self.manifest.match_request("security")
        self.assertTrue(len(matches) > 0)

    def test_no_match_below_threshold(self):
        matches = self.manifest.match_request("unrelated quantum physics", threshold=0.5)
        self.assertEqual(len(matches), 0)

    def test_manifest_properties(self):
        self.assertEqual(self.manifest.total_skills, 1)
        self.assertEqual(self.manifest.total_actions, 2)
        self.assertIn("code", self.manifest.all_tags)

    def test_manifest_hash_deterministic(self):
        h1 = self.manifest._compute_hash()
        h2 = self.manifest._compute_hash()
        self.assertEqual(h1, h2)

    def test_manifest_serialization_roundtrip(self):
        d = self.manifest.to_dict()
        restored = AgentManifest.from_dict(d)
        self.assertEqual(restored.identity.name, "TestAgent")
        self.assertEqual(restored.total_actions, 2)


class TestMessage(unittest.TestCase):
    def test_auto_id_and_timestamp(self):
        msg = Message(from_agent="a1", to_agent="a2", subject="Hello")
        self.assertTrue(len(msg.message_id) > 0)
        self.assertTrue(len(msg.timestamp) > 0)

    def test_expiry(self):
        msg = Message(
            from_agent="a1",
            to_agent="a2",
            timestamp=(datetime.utcnow() - timedelta(hours=2)).isoformat(),
            ttl_seconds=3600,
        )
        self.assertTrue(msg.is_expired)

    def test_not_expired(self):
        msg = Message(from_agent="a1", to_agent="a2", ttl_seconds=3600)
        self.assertFalse(msg.is_expired)

    def test_roundtrip(self):
        msg = Message(from_agent="a1", to_agent="a2", subject="Test", body={"key": "val"})
        restored = Message.from_dict(msg.to_dict())
        self.assertEqual(restored.subject, "Test")
        self.assertEqual(restored.body["key"], "val")


class TestKnowledgeEntry(unittest.TestCase):
    def test_content_hash_dedup(self):
        e1 = KnowledgeEntry(content="Dynamic pricing improves margins by 15%")
        e2 = KnowledgeEntry(content="Dynamic pricing improves margins by 15%")
        self.assertEqual(e1.entry_id, e2.entry_id)

    def test_different_content_different_id(self):
        e1 = KnowledgeEntry(content="Fact A")
        e2 = KnowledgeEntry(content="Fact B")
        self.assertNotEqual(e1.entry_id, e2.entry_id)

    def test_relevance_score(self):
        e = KnowledgeEntry(content="test", confidence=0.7)
        score = e.relevance_score
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)

    def test_endorsement_boosts_score(self):
        e = KnowledgeEntry(content="test", confidence=0.5, endorsements=["a1", "a2", "a3"])
        self.assertGreater(e.relevance_score, 0.5)

    def test_dispute_lowers_score(self):
        e = KnowledgeEntry(content="test", confidence=0.5, disputes=["a1", "a2"])
        self.assertLess(e.relevance_score, 0.5)

    def test_expiry(self):
        e = KnowledgeEntry(
            content="old news",
            expires_at=(datetime.utcnow() - timedelta(days=1)).isoformat(),
        )
        self.assertTrue(e.is_expired)


class TestServiceListing(unittest.TestCase):
    def test_profit_margin(self):
        s = ServiceListing(name="Review", price=0.10, estimated_cost=0.01)
        self.assertAlmostEqual(s.profit_margin, 0.9, places=1)

    def test_zero_price_margin(self):
        s = ServiceListing(name="Free", price=0.0)
        self.assertEqual(s.profit_margin, 0.0)


class TestAgentNetwork(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.identity_a = AgentIdentity("agent_a", "Alice", "ALICE", "coder")
        self.identity_b = AgentIdentity("agent_b", "Bob", "BOB", "writer")
        self.network_a = AgentNetwork(self.identity_a, data_dir=self.tmpdir)
        self.network_b = AgentNetwork(self.identity_b, data_dir=self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ─── Registration & Discovery ─────

    def test_register_and_discover(self):
        self.network_a.register()
        self.network_b.register()

        agents = self.network_a.discover_agents()
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0].identity.name, "Bob")

    def test_discover_by_type(self):
        self.network_a.register()
        self.network_b.register()

        coders = self.network_b.discover_agents(agent_type="coder")
        self.assertEqual(len(coders), 1)
        self.assertEqual(coders[0].identity.name, "Alice")

        writers = self.network_a.discover_agents(agent_type="writer")
        self.assertEqual(len(writers), 1)

    def test_discover_excludes_self(self):
        self.network_a.register()
        agents = self.network_a.discover_agents()
        self.assertEqual(len(agents), 0)

    def test_discover_include_self(self):
        self.network_a.register()
        agents = self.network_a.discover_agents(exclude_self=False)
        self.assertEqual(len(agents), 1)

    def test_heartbeat(self):
        self.network_a.register()
        self.network_a.heartbeat()
        # Should not error

    # ─── Messaging ────────────────────

    def test_send_and_receive_message(self):
        self.network_a.register()
        self.network_b.register()

        msg = Message(to_agent="agent_b", subject="Hello Bob", body={"text": "Hi!"})
        mid = self.network_a.send_message(msg)
        self.assertTrue(len(mid) > 0)

        received = self.network_b.check_messages()
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].subject, "Hello Bob")
        self.assertEqual(received[0].from_agent, "agent_a")

    def test_messages_drained_after_read(self):
        msg = Message(to_agent="agent_b", subject="Test")
        self.network_a.send_message(msg)

        received = self.network_b.check_messages()
        self.assertEqual(len(received), 1)

        # Second read should be empty (drained)
        received2 = self.network_b.check_messages()
        self.assertEqual(len(received2), 0)

    def test_messages_not_drained_when_drain_false(self):
        msg = Message(to_agent="agent_b", subject="Test")
        self.network_a.send_message(msg)

        received = self.network_b.check_messages(drain=False)
        self.assertEqual(len(received), 1)

        received2 = self.network_b.check_messages(drain=False)
        self.assertEqual(len(received2), 1)

    def test_filter_by_message_type(self):
        self.network_a.send_message(Message(
            to_agent="agent_b",
            message_type=MessageType.REQUEST.value,
            subject="Request",
        ))
        self.network_a.send_message(Message(
            to_agent="agent_b",
            message_type=MessageType.BROADCAST.value,
            subject="Broadcast",
        ))

        requests = self.network_b.check_messages(message_type=MessageType.REQUEST.value)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].subject, "Request")

    def test_reply(self):
        self.network_a.send_message(Message(to_agent="agent_b", subject="Question"))
        received = self.network_b.check_messages()
        self.network_b.reply(received[0], {"answer": "42"})

        replies = self.network_a.check_messages()
        self.assertEqual(len(replies), 1)
        self.assertTrue(replies[0].subject.startswith("Re:"))
        self.assertEqual(replies[0].body["answer"], "42")

    def test_broadcast(self):
        self.network_a.register()
        self.network_b.register()

        identity_c = AgentIdentity("agent_c", "Charlie", "CHAR")
        network_c = AgentNetwork(identity_c, data_dir=self.tmpdir)
        network_c.register()

        mids = self.network_a.broadcast("Announcement", {"info": "System upgrade"})
        self.assertEqual(len(mids), 2)  # Bob and Charlie

        bob_msgs = self.network_b.check_messages()
        self.assertEqual(len(bob_msgs), 1)
        self.assertEqual(bob_msgs[0].subject, "Announcement")

    # ─── Marketplace ──────────────────

    def test_publish_and_list_services(self):
        service = ServiceListing(
            name="Code Review",
            description="AI-powered code review",
            price=0.10,
            tags=["code", "review"],
        )
        published = self.network_a.publish_service(service)
        self.assertEqual(published.agent_id, "agent_a")

        services = self.network_b.list_services()
        self.assertEqual(len(services), 1)
        self.assertEqual(services[0].name, "Code Review")

    def test_filter_services_by_tag(self):
        self.network_a.publish_service(ServiceListing(
            name="Code Review", price=0.10, tags=["code"],
        ))
        self.network_a.publish_service(ServiceListing(
            name="Writing", price=0.05, tags=["writing"],
        ))

        code_services = self.network_b.list_services(tags=["code"])
        self.assertEqual(len(code_services), 1)

    def test_filter_services_by_price(self):
        self.network_a.publish_service(ServiceListing(name="Cheap", price=0.01))
        self.network_a.publish_service(ServiceListing(name="Expensive", price=10.0))

        affordable = self.network_b.list_services(max_price=1.0)
        self.assertEqual(len(affordable), 1)
        self.assertEqual(affordable[0].name, "Cheap")

    def test_create_and_fulfill_order(self):
        service = self.network_a.publish_service(ServiceListing(
            name="Review",
            price=0.10,
        ))

        # Bob places an order
        order = self.network_b.create_order(service.service_id, {"code": "print('hi')"})
        self.assertEqual(order.status, OrderStatus.PENDING.value)
        self.assertEqual(order.price_paid, 0.10)

        # Alice checks for order notifications
        msgs = self.network_a.check_messages()
        self.assertEqual(len(msgs), 1)
        self.assertIn("order", msgs[0].subject.lower())

        # Alice fulfills the order
        fulfilled = self.network_a.fulfill_order(
            order.order_id,
            result={"grade": "A", "issues": []},
        )
        self.assertEqual(fulfilled.status, OrderStatus.COMPLETED.value)

        # Bob gets notified
        bob_msgs = self.network_b.check_messages()
        self.assertEqual(len(bob_msgs), 1)
        self.assertIn("completed", bob_msgs[0].subject.lower())

    def test_order_failure(self):
        service = self.network_a.publish_service(ServiceListing(name="Flaky", price=0.05))
        order = self.network_b.create_order(service.service_id)

        # Drain Alice's notification
        self.network_a.check_messages()

        failed = self.network_a.fulfill_order(order.order_id, error="Service unavailable")
        self.assertEqual(failed.status, OrderStatus.FAILED.value)
        self.assertEqual(failed.error, "Service unavailable")

    def test_my_orders(self):
        service = self.network_a.publish_service(ServiceListing(name="Svc", price=0.01))
        self.network_b.create_order(service.service_id)

        bob_orders = self.network_b.my_orders(as_customer=True)
        self.assertEqual(len(bob_orders), 1)

        alice_orders = self.network_a.my_orders(as_customer=False)
        self.assertEqual(len(alice_orders), 1)

    def test_get_order(self):
        service = self.network_a.publish_service(ServiceListing(name="Svc", price=0.01))
        order = self.network_b.create_order(service.service_id)

        retrieved = self.network_b.get_order(order.order_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.order_id, order.order_id)

    # ─── Knowledge Sharing ────────────

    def test_publish_and_query_knowledge(self):
        entry = KnowledgeEntry(
            content="Dynamic pricing increases margins by 15%",
            category=KnowledgeCategory.OPTIMIZATION.value,
            confidence=0.8,
            tags=["pricing", "revenue"],
        )
        self.network_a.publish_knowledge(entry)

        results = self.network_b.query_knowledge(
            category=KnowledgeCategory.OPTIMIZATION.value,
        )
        self.assertEqual(len(results), 1)
        self.assertIn("pricing", results[0].content.lower())

    def test_knowledge_deduplication(self):
        content = "Caching reduces API costs by 40%"
        self.network_a.publish_knowledge(KnowledgeEntry(content=content, confidence=0.6))
        self.network_b.publish_knowledge(KnowledgeEntry(content=content, confidence=0.9))

        results = self.network_a.query_knowledge()
        self.assertEqual(len(results), 1)
        # Higher confidence should win
        self.assertGreaterEqual(results[0].confidence, 0.9)

    def test_knowledge_tag_filter(self):
        self.network_a.publish_knowledge(KnowledgeEntry(
            content="Fact A", tags=["pricing"],
        ))
        self.network_a.publish_knowledge(KnowledgeEntry(
            content="Fact B", tags=["security"],
        ))

        pricing_knowledge = self.network_b.query_knowledge(tags=["pricing"])
        self.assertEqual(len(pricing_knowledge), 1)

    def test_knowledge_text_search(self):
        self.network_a.publish_knowledge(KnowledgeEntry(content="Python is great for data science"))
        self.network_a.publish_knowledge(KnowledgeEntry(content="Rust is fast for systems"))

        results = self.network_b.query_knowledge(search_text="Python")
        self.assertEqual(len(results), 1)

    def test_endorse_knowledge(self):
        entry = KnowledgeEntry(content="Important fact", confidence=0.5)
        self.network_a.publish_knowledge(entry)

        updated = self.network_b.endorse_knowledge(entry.entry_id)
        self.assertIsNotNone(updated)
        self.assertGreater(updated.confidence, 0.5)
        self.assertIn("agent_b", updated.endorsements)

    def test_dispute_knowledge(self):
        entry = KnowledgeEntry(content="Dubious claim", confidence=0.5)
        self.network_a.publish_knowledge(entry)

        updated = self.network_b.dispute_knowledge(entry.entry_id, reason="Inaccurate")
        self.assertIsNotNone(updated)
        self.assertLess(updated.confidence, 0.5)

    def test_double_endorse_noop(self):
        entry = KnowledgeEntry(content="Fact", confidence=0.5)
        self.network_a.publish_knowledge(entry)

        self.network_b.endorse_knowledge(entry.entry_id)
        result1 = self.network_b.endorse_knowledge(entry.entry_id)
        # Should only have one endorsement
        self.assertEqual(len(result1.endorsements), 1)

    # ─── Stats & Cleanup ─────────────

    def test_my_stats(self):
        self.network_a.register()
        self.network_a.publish_service(ServiceListing(name="Svc", price=0.01))

        stats = self.network_a.my_stats()
        self.assertEqual(stats["name"], "Alice")
        self.assertEqual(stats["services_published"], 1)

    def test_cleanup_expired(self):
        # Add an expired message
        from adam_toolkit.agent_protocol import JsonStore
        msg_store = JsonStore(os.path.join(self.tmpdir, "messages.json"))
        msg_store.save({
            "agent_a": [
                Message(
                    from_agent="agent_b",
                    to_agent="agent_a",
                    subject="Old msg",
                    timestamp=(datetime.utcnow() - timedelta(hours=2)).isoformat(),
                    ttl_seconds=3600,
                ).to_dict(),
            ]
        })

        self.network_a.cleanup_expired()
        msgs = self.network_a.check_messages()
        self.assertEqual(len(msgs), 0)

    def test_find_agent_for_task(self):
        manifest_a = AgentManifest(
            identity=self.identity_a,
            capabilities=[
                CapabilityGroup(
                    skill_id="code",
                    name="Code",
                    description="Coding tasks",
                    actions=[
                        Capability(
                            name="code_review",
                            description="Review code for bugs",
                            tags=["code", "review", "bugs"],
                        ),
                    ],
                ),
            ],
        )
        self.network_a.register(manifest_a)

        results = self.network_b.find_agent_for_task("review code bugs")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0][0].identity.name, "Alice")


class TestServiceOrder(unittest.TestCase):
    def test_auto_id(self):
        order = ServiceOrder(service_id="svc_1", customer_agent_id="a1")
        self.assertTrue(len(order.order_id) > 0)

    def test_roundtrip(self):
        order = ServiceOrder(
            service_id="svc_1",
            customer_agent_id="a1",
            provider_agent_id="a2",
            params={"key": "val"},
            price_paid=0.10,
        )
        restored = ServiceOrder.from_dict(order.to_dict())
        self.assertEqual(restored.service_id, "svc_1")
        self.assertEqual(restored.price_paid, 0.10)


if __name__ == "__main__":
    unittest.main()
