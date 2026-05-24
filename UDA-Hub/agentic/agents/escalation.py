"""Escalation Agent - Handles cases that cannot be auto-resolved."""
from agentic.tools.ticket_tools import update_ticket_status, log_ticket_message


ESCALATION_SYSTEM_PROMPT = """You are an Escalation Agent for CultPass customer support.

Your job is to handle cases that cannot be automatically resolved and prepare them for human agents.

**When you are invoked, it means:**
- The Resolver Agent could not confidently resolve the issue
- The issue requires human judgment (e.g., refunds, account unblocking, fraud)
- The customer has expressed dissatisfaction with automated responses

**Your responsibilities:**
1. Summarize the issue clearly for the human support team
2. Document what has been attempted so far
3. Set appropriate priority level
4. Provide a professional response to the customer
5. Log the escalation in the ticket system if a ticket_id is available

**Escalation Priority Levels:**
- P1 (Critical): Security issues, active fraud, system outages
- P2 (High): Blocked accounts, payment failures, urgent event issues
- P3 (Medium): Refund requests, subscription disputes, quality complaints
- P4 (Low): Feature requests, general feedback, minor inconveniences

**Response Format:**
ESCALATION:
Priority: [P1/P2/P3/P4]
Summary: [Clear summary for human agent]
Attempted: [What was tried before escalation]
Customer Response: [Professional message to the customer acknowledging escalation]
"""


def build_escalation(message: str, classification: str, urgency: str, ticket_id: str | None = None) -> dict:
    """Create a deterministic escalation package and persist status when possible."""
    priority_map = {"critical": "P1", "high": "P2", "medium": "P3", "low": "P4"}
    priority = priority_map.get(urgency, "P3")
    attempted = "Knowledge lookup and deterministic routing completed. Human review required."
    customer_response = (
        f"Your case has been escalated with {priority} priority. "
        f"A human support specialist will review your {classification} issue and follow up shortly."
    )

    actions = []
    if ticket_id:
        actions.append(update_ticket_status.invoke({"ticket_id": ticket_id, "new_status": "escalated", "issue_type": classification}))
        actions.append(log_ticket_message.invoke({"ticket_id": ticket_id, "role": "system", "content": f"Escalated with {priority} priority: {message}"}))

    return {
        "priority": priority,
        "summary": f"{classification.title()} issue: {message[:160]}",
        "attempted": attempted,
        "customer_response": customer_response,
        "actions": actions,
    }


escalation_agent = {
    "name": "escalation",
    "system_prompt": ESCALATION_SYSTEM_PROMPT,
    "run": build_escalation,
}
