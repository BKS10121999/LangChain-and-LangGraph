"""
Test cases for UDA-Hub multi-agent system.

Covers:
1. Database setup verification
2. Knowledge retrieval (RAG)
3. Account tools
4. Ticket tools
5. Workflow routing logic
6. End-to-end ticket processing
7. Memory / state management
8. Knowledge base diversity

Run with: python -m pytest tests/ -v
Or:       python tests/test_agents.py
"""
import os
import sys
import json
import uuid
import pytest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from data.models import cultpass, udahub


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CULTPASS_DB = os.path.join(PROJECT_ROOT, "data", "external", "cultpass.db")
UDAHUB_DB = os.path.join(PROJECT_ROOT, "data", "core", "udahub.db")


def db_exists(path):
    return os.path.exists(path)


# ============================================================
# Test 1: Database Setup Verification
# ============================================================

class TestDatabaseSetup:
    """Verify databases are properly initialized."""

    def test_cultpass_db_exists(self):
        assert db_exists(CULTPASS_DB), f"CultPass DB not found. Run 01_external_db_setup.ipynb first."

    def test_udahub_db_exists(self):
        assert db_exists(UDAHUB_DB), f"UDA-Hub DB not found. Run 02_core_db_setup.ipynb first."

    def test_cultpass_has_users(self):
        if not db_exists(CULTPASS_DB):
            pytest.skip("CultPass DB not set up yet")
        engine = create_engine(f"sqlite:///{CULTPASS_DB}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            users = session.query(cultpass.User).all()
            assert len(users) >= 6, f"Expected >= 6 users, got {len(users)}"
        finally:
            session.close()

    def test_cultpass_has_experiences(self):
        if not db_exists(CULTPASS_DB):
            pytest.skip("CultPass DB not set up yet")
        engine = create_engine(f"sqlite:///{CULTPASS_DB}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            experiences = session.query(cultpass.Experience).all()
            assert len(experiences) >= 7, f"Expected >= 7 experiences, got {len(experiences)}"
        finally:
            session.close()

    def test_cultpass_has_subscriptions(self):
        if not db_exists(CULTPASS_DB):
            pytest.skip("CultPass DB not set up yet")
        engine = create_engine(f"sqlite:///{CULTPASS_DB}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            subs = session.query(cultpass.Subscription).all()
            assert len(subs) >= 6, f"Expected >= 6 subscriptions, got {len(subs)}"
        finally:
            session.close()

    def test_udahub_has_knowledge_articles(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        engine = create_engine(f"sqlite:///{UDAHUB_DB}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            articles = session.query(udahub.Knowledge).all()
            assert len(articles) >= 14, f"Expected >= 14 articles, got {len(articles)}"
        finally:
            session.close()

    def test_udahub_has_account(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        engine = create_engine(f"sqlite:///{UDAHUB_DB}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            account = session.query(udahub.Account).filter_by(account_id="cultpass").first()
            assert account is not None
            assert account.account_name == "CultPass Card"
        finally:
            session.close()

    def test_udahub_has_tickets(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        engine = create_engine(f"sqlite:///{UDAHUB_DB}", echo=False)
        Session = sessionmaker(bind=engine)
        session = Session()
        try:
            tickets = session.query(udahub.Ticket).all()
            assert len(tickets) >= 1, f"Expected >= 1 ticket, got {len(tickets)}"
        finally:
            session.close()


# ============================================================
# Test 2: Knowledge Retrieval (RAG)
# ============================================================

class TestKnowledgeRetrieval:
    """Test knowledge base search functionality."""

    def test_search_login_issue(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.tools.knowledge_tools import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "login password reset", "account_id": "cultpass"})
        assert "login" in result.lower() or "password" in result.lower()
        assert "NO_RELEVANT_ARTICLES_FOUND" not in result

    def test_search_subscription(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.tools.knowledge_tools import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "cancel subscription plan", "account_id": "cultpass"})
        assert "subscription" in result.lower() or "cancel" in result.lower()

    def test_search_refund(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.tools.knowledge_tools import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "refund payment", "account_id": "cultpass"})
        assert "refund" in result.lower()

    def test_search_no_results(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.tools.knowledge_tools import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "xyzzy foobar baz quantum", "account_id": "cultpass"})
        assert "NO_RELEVANT_ARTICLES_FOUND" in result

    def test_search_reservation(self):
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.tools.knowledge_tools import search_knowledge_base
        result = search_knowledge_base.invoke({"query": "reserve event booking", "account_id": "cultpass"})
        assert "reserve" in result.lower() or "booking" in result.lower()


# ============================================================
# Test 3: Account Tools
# ============================================================

class TestAccountTools:
    """Test account lookup and management tools."""

    def test_lookup_user_by_email(self):
        if not db_exists(CULTPASS_DB):
            pytest.skip("CultPass DB not set up yet")
        from agentic.tools.account_tools import lookup_user
        result = lookup_user.invoke({"email": "alice.kingsley@wonderland.com"})
        assert "Alice" in result or "alice" in result.lower()

    def test_lookup_user_not_found(self):
        if not db_exists(CULTPASS_DB):
            pytest.skip("CultPass DB not set up yet")
        from agentic.tools.account_tools import lookup_user
        result = lookup_user.invoke({"email": "nonexistent@nowhere.com"})
        assert "No user found" in result

    def test_get_subscription(self):
        if not db_exists(CULTPASS_DB):
            pytest.skip("CultPass DB not set up yet")
        from agentic.tools.account_tools import get_subscription_details
        result = get_subscription_details.invoke({"user_id": "a4ab87"})
        assert "tier" in result.lower()
        assert "status" in result.lower()

    def test_check_reservations(self):
        if not db_exists(CULTPASS_DB):
            pytest.skip("CultPass DB not set up yet")
        from agentic.tools.account_tools import check_user_reservations
        result = check_user_reservations.invoke({"user_id": "a4ab87"})
        assert "reservation" in result.lower() or "No reservations" in result


# ============================================================
# Test 4: Ticket Tools
# ============================================================

class TestTicketTools:
    """Test ticket management tools."""

    def test_update_ticket_invalid_status(self):
        from agentic.tools.ticket_tools import update_ticket_status
        result = update_ticket_status.invoke({"ticket_id": "fake-id", "new_status": "invalid_status"})
        assert "Invalid status" in result

    def test_log_message_invalid_role(self):
        from agentic.tools.ticket_tools import log_ticket_message
        result = log_ticket_message.invoke({"ticket_id": "fake-id", "role": "invalid_role", "content": "test"})
        assert "Invalid role" in result


# ============================================================
# Test 5: Workflow Routing Logic
# ============================================================

class TestWorkflowRouting:
    """Test the supervisor routing decisions."""

    def test_workflow_compiles(self):
        from agentic.workflow import build_workflow
        workflow = build_workflow()
        assert workflow is not None

    def test_workflow_has_nodes(self):
        from agentic.workflow import build_workflow
        workflow = build_workflow()
        node_names = list(workflow.get_graph().nodes.keys())
        assert "supervisor" in node_names
        assert "classifier" in node_names
        assert "resolver" in node_names
        assert "escalation" in node_names
        assert "memory" in node_names

    def test_supervisor_routes_to_classifier_initially(self):
        from agentic.workflow import supervisor_node
        state = {
            "messages": [HumanMessage(content="I need help")],
            "classification": "",
            "urgency": "",
            "next_agent": "",
            "resolution_confidence": "",
            "escalation_reason": "",
        }
        result = supervisor_node(state)
        assert result["next_agent"] == "classifier"

    def test_supervisor_routes_to_escalation_on_critical(self):
        from agentic.workflow import supervisor_node
        state = {
            "messages": [HumanMessage(content="Security breach!")],
            "classification": "account",
            "urgency": "critical",
            "next_agent": "",
            "resolution_confidence": "",
            "escalation_reason": "",
        }
        result = supervisor_node(state)
        assert result["next_agent"] == "escalation"

    def test_supervisor_routes_to_resolver_after_classification(self):
        from agentic.workflow import supervisor_node
        state = {
            "messages": [HumanMessage(content="How do I cancel?")],
            "classification": "subscription",
            "urgency": "medium",
            "next_agent": "",
            "resolution_confidence": "",
            "escalation_reason": "",
        }
        result = supervisor_node(state)
        assert result["next_agent"] == "resolver"

    def test_supervisor_ends_on_high_confidence(self):
        from agentic.workflow import supervisor_node
        state = {
            "messages": [HumanMessage(content="thanks")],
            "classification": "general",
            "urgency": "low",
            "next_agent": "",
            "resolution_confidence": "HIGH",
            "escalation_reason": "",
        }
        result = supervisor_node(state)
        assert result["next_agent"] == "END"

    def test_supervisor_escalates_on_low_confidence(self):
        from agentic.workflow import supervisor_node
        state = {
            "messages": [HumanMessage(content="complex issue")],
            "classification": "technical",
            "urgency": "high",
            "next_agent": "",
            "resolution_confidence": "LOW",
            "escalation_reason": "",
        }
        result = supervisor_node(state)
        assert result["next_agent"] == "escalation"


# ============================================================
# Test 6: End-to-End Ticket Processing
# ============================================================

class TestEndToEnd:
    """Test complete ticket processing flow."""

    def test_simple_query_resolution(self):
        """Test E2E for a simple support query."""
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.workflow import orchestrator

        config = {"configurable": {"thread_id": "test-e2e-simple"}}
        result = orchestrator.invoke(
            {"messages": [HumanMessage(content="How do I reserve an event on CultPass?")]},
            config=config
        )
        assert len(result["messages"]) > 1
        last_msg = result["messages"][-1].content
        assert len(last_msg) > 10

    def test_escalation_flow(self):
        """Test E2E for a critical issue that triggers escalation."""
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.workflow import orchestrator

        config = {"configurable": {"thread_id": "test-e2e-escalate"}}
        result = orchestrator.invoke(
            {"messages": [HumanMessage(content="My account has been hacked and someone is using my credit card!")]},
            config=config
        )
        assert len(result["messages"]) > 1
        last_msg = result["messages"][-1].content
        assert len(last_msg) > 10


# ============================================================
# Test 7: Memory & State Management
# ============================================================

class TestMemory:
    """Test session memory and state inspection."""

    def test_session_memory_persists(self):
        """Short-term memory persists within thread_id."""
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.workflow import orchestrator

        thread_id = "test-memory-persist"
        config = {"configurable": {"thread_id": thread_id}}

        result1 = orchestrator.invoke(
            {"messages": [HumanMessage(content="Hi, I need help with my subscription")]},
            config=config
        )
        assert len(result1["messages"]) > 1

        result2 = orchestrator.invoke(
            {"messages": [HumanMessage(content="What tier am I on?")]},
            config=config
        )
        # Should have more messages accumulated
        assert len(result2["messages"]) > len(result1["messages"])

    def test_state_inspection_via_thread(self):
        """Verify state history is accessible via thread_id."""
        if not db_exists(UDAHUB_DB):
            pytest.skip("UDA-Hub DB not set up yet")
        from agentic.workflow import orchestrator

        thread_id = "test-state-inspect"
        config = {"configurable": {"thread_id": thread_id}}

        orchestrator.invoke(
            {"messages": [HumanMessage(content="How do I cancel my subscription?")]},
            config=config
        )

        state_history = list(orchestrator.get_state_history(config))
        assert len(state_history) > 0
        latest = state_history[0]
        assert "messages" in latest.values
        assert len(latest.values["messages"]) > 0


# ============================================================
# Test 8: Knowledge Base Article Diversity
# ============================================================

class TestKnowledgeBaseDiversity:
    """Verify articles cover diverse topics."""

    def test_at_least_14_articles(self):
        articles_path = os.path.join(PROJECT_ROOT, "data", "external", "cultpass_articles.jsonl")
        articles = []
        with open(articles_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))
        assert len(articles) >= 14

    def test_covers_multiple_categories(self):
        articles_path = os.path.join(PROJECT_ROOT, "data", "external", "cultpass_articles.jsonl")
        articles = []
        with open(articles_path, "r", encoding="utf-8") as f:
            for line in f:
                articles.append(json.loads(line))

        all_tags = set()
        for article in articles:
            tags = [t.strip() for t in article["tags"].split(",")]
            all_tags.update(tags)

        category_keywords = {
            "billing": ["billing", "payment", "refund", "pricing"],
            "subscription": ["subscription", "tier", "premium", "basic"],
            "technical": ["technical", "app", "crash", "bug", "troubleshooting"],
            "account": ["account", "blocked", "login", "password", "privacy"],
            "reservation": ["reservation", "booking", "events", "attendance"],
        }

        covered = set()
        for category, keywords in category_keywords.items():
            for kw in keywords:
                if kw in all_tags:
                    covered.add(category)
                    break

        assert len(covered) >= 4, f"Only covers {covered}, expected >= 4 categories"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
