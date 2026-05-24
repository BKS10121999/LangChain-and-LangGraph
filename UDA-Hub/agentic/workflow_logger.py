"""Structured Workflow Logger for UDA-Hub Agentic System.

Persists structured, searchable log events as JSONL records.
Each event captures: event_type, ticket_id, agent, decision, route,
tool_name, tool_result_summary, confidence, final_action, and timestamp.
"""
import json
import os
from datetime import datetime, timezone

LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(LOGS_DIR, "workflow_events.jsonl")


def _ensure_log_dir():
    os.makedirs(LOGS_DIR, exist_ok=True)


def log_event(
    event_type: str,
    agent: str,
    *,
    ticket_id: str = "",
    thread_id: str = "",
    decision: str = "",
    route: str = "",
    tool_name: str = "",
    tool_result_summary: str = "",
    confidence: str = "",
    final_action: str = "",
    category: str = "",
    urgency: str = "",
    extras: dict | None = None,
) -> dict:
    """Write a structured log event to the JSONL log file and return it."""
    _ensure_log_dir()
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "agent": agent,
        "ticket_id": ticket_id,
        "thread_id": thread_id,
        "decision": decision,
        "route": route,
        "tool_name": tool_name,
        "tool_result_summary": tool_result_summary,
        "confidence": confidence,
        "final_action": final_action,
        "category": category,
        "urgency": urgency,
    }
    if extras:
        event["extras"] = extras

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

    return event


def get_logs(thread_id: str | None = None, event_type: str | None = None) -> list[dict]:
    """Read and optionally filter structured log events."""
    if not os.path.exists(LOG_FILE):
        return []

    events = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if thread_id and event.get("thread_id") != thread_id:
                continue
            if event_type and event.get("event_type") != event_type:
                continue
            events.append(event)
    return events


def search_logs(**filters) -> list[dict]:
    """Return log events that match all provided exact-match filters."""
    events = get_logs()
    if not filters:
        return events

    matched_events = []
    for event in events:
        if all(event.get(key) == value for key, value in filters.items()):
            matched_events.append(event)
    return matched_events


def clear_logs():
    """Clear all workflow log events (useful for testing)."""
    _ensure_log_dir()
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")


def get_logs_summary(thread_id: str | None = None) -> str:
    """Return a formatted summary of workflow events for display."""
    events = get_logs(thread_id=thread_id)
    if not events:
        return "No workflow events found."

    lines = [f"{'='*70}", f"STRUCTURED WORKFLOW LOG — {len(events)} event(s)", f"{'='*70}"]
    for i, event in enumerate(events, 1):
        lines.append(f"\n[{i}] {event['timestamp']}")
        lines.append(f"    Event Type : {event['event_type']}")
        lines.append(f"    Agent      : {event['agent']}")
        if event.get("ticket_id"):
            lines.append(f"    Ticket ID  : {event['ticket_id']}")
        if event.get("thread_id"):
            lines.append(f"    Thread ID  : {event['thread_id']}")
        if event.get("decision"):
            lines.append(f"    Decision   : {event['decision']}")
        if event.get("route"):
            lines.append(f"    Route      : {event['route']}")
        if event.get("tool_name"):
            lines.append(f"    Tool       : {event['tool_name']}")
        if event.get("tool_result_summary"):
            lines.append(f"    Tool Result: {event['tool_result_summary']}")
        if event.get("confidence"):
            lines.append(f"    Confidence : {event['confidence']}")
        if event.get("category"):
            lines.append(f"    Category   : {event['category']}")
        if event.get("urgency"):
            lines.append(f"    Urgency    : {event['urgency']}")
        if event.get("final_action"):
            lines.append(f"    Final Act  : {event['final_action']}")
    lines.append(f"\n{'='*70}")
    return "\n".join(lines)
