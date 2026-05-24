"""Classifier Agent - Analyzes and classifies incoming support tickets."""
import re


CLASSIFIER_SYSTEM_PROMPT = """You are a Ticket Classifier Agent for CultPass customer support.

Your job is to analyze incoming customer messages and classify them accurately.

For each message, determine:
1. **Category** (one of): billing, subscription, technical, account, reservation, general
2. **Urgency** (one of): low, medium, high, critical
3. **Key Entities**: Extract any user IDs, emails, ticket IDs, or specific issues mentioned

Classification Rules:
- billing: payment issues, refunds, charges, invoices
- subscription: plan changes, upgrades, cancellations, quota questions
- technical: app crashes, login issues, bugs, performance problems
- account: blocked accounts, profile updates, data privacy, deletion
- reservation: booking, cancellations, event questions, QR codes
- general: FAQ, feature questions, how-to guides

Urgency Rules:
- critical: blocked accounts, security issues, active event problems
- high: login issues, payment failures, upcoming event problems
- medium: subscription questions, reservation changes, complaints
- low: general questions, feature inquiries, feedback

Always respond with a structured classification in this format:
CLASSIFICATION:
- Category: [category]
- Urgency: [urgency]
- Key Entities: [entities or "none"]
- Summary: [brief one-line summary of the issue]
- Recommended Action: [resolve/escalate/need_more_info]
"""

CATEGORY_KEYWORDS = {
    "billing": {"billing", "payment", "refund", "charge", "charged", "invoice", "credit", "card"},
    "subscription": {"subscription", "plan", "tier", "upgrade", "downgrade", "pause", "cancel"},
    "technical": {"technical", "crash", "bug", "error", "login", "password", "qr", "app", "slow"},
    "account": {"account", "blocked", "privacy", "delete", "deletion", "profile", "hacked", "fraud"},
    "reservation": {"reservation", "reserve", "booking", "book", "event", "guest", "waitlist", "qr"},
}

CRITICAL_KEYWORDS = {"hacked", "fraud", "stolen", "breach", "credit card", "security"}
HIGH_KEYWORDS = {"blocked", "payment failed", "cannot login", "can't log in", "urgent", "today", "tomorrow"}


def _tokenize(message: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", (message or "").lower()))


def classify_ticket(message: str) -> dict:
    """Deterministically classify a support message for routing."""
    lowered = (message or "").lower()
    tokens = _tokenize(message)

    best_category = "general"
    best_score = 0
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in tokens or keyword in lowered)
        if score > best_score:
            best_category = category
            best_score = score

    urgency = "low"
    if any(keyword in lowered for keyword in CRITICAL_KEYWORDS):
        urgency = "critical"
    elif any(keyword in lowered for keyword in HIGH_KEYWORDS):
        urgency = "high"
    elif best_category in {"billing", "technical", "reservation", "subscription", "account"}:
        urgency = "medium"

    entities = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", message or "")
    summary = (message or "").strip().replace("\n", " ")[:120]
    recommended_action = "escalate" if urgency == "critical" else ("need_more_info" if best_category == "general" else "resolve")

    return {
        "category": best_category,
        "urgency": urgency,
        "entities": entities or ["none"],
        "summary": summary,
        "recommended_action": recommended_action,
    }


classifier_agent = {
    "name": "classifier",
    "system_prompt": CLASSIFIER_SYSTEM_PROMPT,
    "run": classify_ticket,
}
