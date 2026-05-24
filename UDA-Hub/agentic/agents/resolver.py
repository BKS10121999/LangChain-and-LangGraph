"""Resolver Agent - Resolves tickets using knowledge base and tools."""
import re
from agentic.tools.knowledge_tools import search_knowledge_base, search_knowledge_records
from agentic.tools.account_tools import (
    lookup_user,
    lookup_user_record,
    get_subscription_details,
    check_user_reservations,
)


RESOLVER_SYSTEM_PROMPT = """You are a Resolution Agent for CultPass customer support.

Your job is to resolve customer support tickets using the available tools and knowledge base.

**Workflow:**
1. First, search the knowledge base for relevant articles about the customer's issue
2. If the issue involves account-specific data (subscription, reservations, blocked status), use the account tools
3. Provide a clear, helpful response based on the retrieved information
4. If you cannot resolve the issue with available information, clearly state that escalation is needed

**Tools Available:**
- search_knowledge_base: Search for relevant support articles
- lookup_user: Look up user details by email
- get_subscription_details: Get subscription info by user_id
- check_user_reservations: Get reservation history by user_id

**Response Guidelines:**
- Always base your response on knowledge base articles when available
- Be empathetic and professional
- Provide step-by-step instructions when applicable
- If the knowledge base returns "NO_RELEVANT_ARTICLES_FOUND", recommend escalation
- Include confidence level in your response: HIGH (clear match), MEDIUM (partial match), LOW (escalate)

**Format your response as:**
RESOLUTION:
Confidence: [HIGH/MEDIUM/LOW]
Response: [Your helpful response to the customer]
Actions Taken: [List of tools used and their results]
"""


def _extract_email(message: str) -> str | None:
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", message or "")
    return match.group(0) if match else None


def resolve_ticket(message: str, classification: str, known_user: dict | None = None) -> dict:
    """Resolve a ticket using the local tools and deterministic confidence rules."""
    actions = []
    knowledge_records = search_knowledge_records(query=message, account_id="cultpass", limit=2)
    knowledge_text = search_knowledge_base.invoke({"query": message, "account_id": "cultpass"})
    actions.append({"tool": "search_knowledge_base", "result": knowledge_text})

    email = _extract_email(message)
    user_record = known_user
    if email and not user_record:
        user_record = lookup_user_record(email=email)
        actions.append({"tool": "lookup_user", "result": lookup_user.invoke({"email": email})})

    account_details = []
    if user_record and classification in {"subscription", "billing", "account", "reservation"}:
        if user_record.get("subscription"):
            account_details.append(get_subscription_details.invoke({"user_id": user_record["user_id"]}))
            actions.append({"tool": "get_subscription_details", "result": account_details[-1]})
        if classification == "reservation":
            reservation_details = check_user_reservations.invoke({"user_id": user_record["user_id"]})
            account_details.append(reservation_details)
            actions.append({"tool": "check_user_reservations", "result": reservation_details})

    if "NO_RELEVANT_ARTICLES_FOUND" in knowledge_text:
        confidence = "LOW"
        response = (
            "I could not find a reliable knowledge base article for this request. "
            "I recommend escalating it to a human support specialist so they can investigate further."
        )
    else:
        confidence = "HIGH" if knowledge_records and knowledge_records[0]["relevance_score"] >= 0.3 else "MEDIUM"
        top_titles = ", ".join(article["title"] for article in knowledge_records) if knowledge_records else "our support guidance"
        response_lines = [
            f"I found relevant guidance in {top_titles}.",
            knowledge_records[0]["content"] if knowledge_records else knowledge_text,
        ]
        if user_record:
            response_lines.append(
                f"I also found your account profile for {user_record['full_name']} ({user_record['email']})."
            )
            if user_record.get("subscription"):
                subscription = user_record["subscription"]
                response_lines.append(
                    f"Current subscription: {subscription['tier']} tier with status {subscription['status']}."
                )
        response = "\n\n".join(response_lines)

    return {
        "confidence": confidence,
        "response": response,
        "actions": actions,
        "used_user_record": user_record,
    }


resolver_agent = {
    "name": "resolver",
    "system_prompt": RESOLVER_SYSTEM_PROMPT,
    "run": resolve_ticket,
}
