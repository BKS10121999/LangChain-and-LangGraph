"""
UDA-Hub Orchestrator Workflow

Multi-agent workflow using LangGraph's StateGraph with a supervisor pattern.
Routes incoming tickets to deterministic specialized agents for repeatable local runs.
"""
import json
import operator
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agentic.agents.classifier import classify_ticket
from agentic.agents.resolver import resolve_ticket
from agentic.agents.escalation import build_escalation
from agentic.agents.memory_agent import resolve_customer_context
from agentic.tools.account_tools import lookup_user_record
from agentic.tools.ticket_tools import (
    fetch_ticket_messages,
    get_latest_ticket_for_external_user,
    log_ticket_message,
    update_ticket_status,
)


# ============================================================
# State Definition
# ============================================================

class AgentState(TypedDict):
    """State shared across all agents in the workflow."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    classification: str
    urgency: str
    next_agent: str
    resolution_confidence: str
    escalation_reason: str
    ticket_id: str
    external_user_id: str
    user_name: str
    memory_summary: str
    memory_checked: bool
    tool_usage: Annotated[list[str], operator.add]
    audit_log: Annotated[list[str], operator.add]


# ============================================================
# Agent Node Functions
# ============================================================

KNOWN_USERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "external",
    "cultpass_users.jsonl",
)


def _load_known_users() -> list[dict]:
    with open(KNOWN_USERS_PATH, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _last_human_message(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _append_audit(state: AgentState, entry: str) -> list[str]:
    return [entry]

def supervisor_node(state: AgentState) -> dict:
    """Supervisor node that routes to the appropriate agent."""
    classification = state.get("classification", "")
    resolution_confidence = state.get("resolution_confidence", "")

    # If we already have a resolution with HIGH confidence, we're done
    if resolution_confidence == "HIGH":
        return {"next_agent": "END", "audit_log": _append_audit(state, "Supervisor ended with HIGH confidence")}

    if resolution_confidence == "MEDIUM":
        return {"next_agent": "END", "audit_log": _append_audit(state, "Supervisor ended with MEDIUM confidence")}

    # If resolution confidence is LOW, escalate
    if resolution_confidence == "LOW":
        return {"next_agent": "escalation", "audit_log": _append_audit(state, "Supervisor routed to escalation on LOW confidence")}

    # If not yet classified, classify first
    if not classification:
        return {"next_agent": "classifier", "audit_log": _append_audit(state, "Supervisor routed to classifier")}

    # After classification, check if escalation is needed
    urgency = state.get("urgency", "")
    if urgency == "critical":
        return {"next_agent": "escalation", "audit_log": _append_audit(state, "Supervisor routed to escalation on critical urgency")}

    # Route to resolver
    return {"next_agent": "resolver", "audit_log": _append_audit(state, "Supervisor routed to resolver")}


def classifier_node(state: AgentState) -> dict:
    """Classifier agent that categorizes the ticket."""
    last_user_message = _last_human_message(state["messages"])
    classification_data = classify_ticket(last_user_message)
    category = classification_data.get("category", "general")
    urgency = classification_data.get("urgency", "medium")

    return {
        "classification": category,
        "urgency": urgency,
        "next_agent": "supervisor",
        "messages": [AIMessage(content=f"[Classifier] Category: {category}, Urgency: {urgency}")],
        "audit_log": _append_audit(state, f"Classifier labeled ticket as {category}/{urgency}"),
    }


def resolver_node(state: AgentState) -> dict:
    """Resolver agent that uses knowledge base and tools to resolve issues."""
    classification = state.get("classification", "general")
    ticket_id = state.get("ticket_id", "")
    last_user_message = _last_human_message(state["messages"])
    known_user = None
    external_user_id = state.get("external_user_id", "")
    if external_user_id:
        known_user = lookup_user_record(user_id=external_user_id)

    result = resolve_ticket(last_user_message, classification, known_user=known_user)
    confidence = result["confidence"]
    response_content = result["response"]
    tool_usage = [action["tool"] for action in result["actions"]]

    if ticket_id:
        log_ticket_message.invoke({"ticket_id": ticket_id, "role": "user", "content": last_user_message})
        log_ticket_message.invoke({"ticket_id": ticket_id, "role": "ai", "content": response_content})
        if confidence in {"HIGH", "MEDIUM"}:
            update_ticket_status.invoke({"ticket_id": ticket_id, "new_status": "resolved", "issue_type": classification})

    return {
        "messages": [AIMessage(content=response_content)],
        "resolution_confidence": confidence,
        "next_agent": "supervisor",
        "tool_usage": tool_usage,
        "audit_log": _append_audit(state, f"Resolver completed with {confidence} confidence"),
    }


def escalation_node(state: AgentState) -> dict:
    """Escalation agent that handles unresolvable cases."""
    classification = state.get("classification", "unknown")
    urgency = state.get("urgency", "medium")
    ticket_id = state.get("ticket_id", "") or None
    last_user_message = _last_human_message(state["messages"])
    escalation = build_escalation(last_user_message, classification, urgency, ticket_id=ticket_id)
    escalation_response = (
        f"I understand your concern, and I want to make sure you get the best possible help.\n\n"
        f"I've escalated your case to our specialized support team with {escalation['priority']} priority.\n\n"
        f"Summary: {escalation['summary']}\n\n"
        f"{escalation['customer_response']}"
    )

    return {
        "messages": [AIMessage(content=escalation_response)],
        "resolution_confidence": "HIGH",  # Mark as done (escalation is a valid resolution)
        "escalation_reason": f"Priority {escalation['priority']}: {classification} issue requiring human review",
        "next_agent": "END",
        "audit_log": _append_audit(state, f"Escalation created at {escalation['priority']} priority"),
    }


def memory_node(state: AgentState) -> dict:
    """Memory agent that retrieves/stores customer context.
    
    Also resets classification and resolution_confidence so the supervisor
    processes the new message from scratch instead of short-circuiting on
    stale state from the previous turn.
    """
    last_user_message = _last_human_message(state["messages"])
    known_users = _load_known_users()
    memory = resolve_customer_context(last_user_message, known_users=known_users)
    matched_user = memory.get("matched_user") or {}
    external_user_id = matched_user.get("id", "")
    user_name = matched_user.get("name", "")
    memory_summary = "No prior customer history found."
    ticket_id = state.get("ticket_id", "")

    if external_user_id:
        latest_ticket = get_latest_ticket_for_external_user(external_user_id)
        if latest_ticket:
            ticket_id = latest_ticket["ticket_id"]
            history = fetch_ticket_messages(ticket_id)
            memory_summary = f"Found {len(history)} prior message(s) for {latest_ticket['user_name']}."
        elif memory.get("history"):
            memory_summary = f"Found customer history for {user_name}."

    if memory.get("history") and memory_summary == "No prior customer history found.":
        memory_summary = f"Retrieved past support history for {user_name}."

    memory_context = f"[Memory] {memory_summary}"

    return {
        "messages": [AIMessage(content=memory_context)],
        "next_agent": "supervisor",
        "ticket_id": ticket_id,
        "external_user_id": external_user_id,
        "user_name": user_name,
        "memory_summary": memory_summary,
        "memory_checked": True,
        # Reset per-turn fields so supervisor re-processes this new message
        "classification": "",
        "resolution_confidence": "",
        "escalation_reason": "",
        "audit_log": _append_audit(state, "Memory agent retrieved customer context"),
    }


# ============================================================
# Routing Logic
# ============================================================

def route_from_supervisor(state: AgentState) -> str:
    """Route from supervisor to the next agent."""
    next_agent = state.get("next_agent", "classifier")
    if next_agent == "END":
        return END
    return next_agent


# ============================================================
# Build the Graph
# ============================================================

def build_workflow():
    """Build and compile the multi-agent workflow graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("resolver", resolver_node)
    workflow.add_node("escalation", escalation_node)
    workflow.add_node("memory", memory_node)

    # Set entry point
    workflow.set_entry_point("memory")

    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "classifier": "classifier",
            "resolver": "resolver",
            "escalation": "escalation",
            "memory": "memory",
            END: END,
        }
    )

    # All agents route back to supervisor
    workflow.add_edge("classifier", "supervisor")
    workflow.add_edge("resolver", "supervisor")
    workflow.add_edge("escalation", "supervisor")
    workflow.add_edge("memory", "supervisor")

    # Compile with memory checkpointer for session management
    checkpointer = MemorySaver()
    compiled = workflow.compile(checkpointer=checkpointer)

    return compiled


# Create the orchestrator instance
orchestrator = build_workflow()