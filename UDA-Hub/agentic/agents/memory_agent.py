"""Memory Agent - Manages long-term memory for personalized support."""
import os
import json
import re
from langchain_core.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import uuid


UDAHUB_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "core", "udahub.db")


def _get_udahub_session():
    """Get a database session for UDA-Hub."""
    engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


def _extract_name_fragment(message: str) -> str | None:
    match = re.search(r"(?:i am|i'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", message or "", re.IGNORECASE)
    if not match:
        return None
    return match.group(1)


@tool
def retrieve_customer_history(user_name: str) -> str:
    """Retrieve past ticket history for a customer by their name. Returns previous interactions and resolutions for personalized support."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import User, Ticket, TicketMessage, TicketMetadata

    session = _get_udahub_session()
    try:
        users = session.query(User).filter(User.user_name.ilike(f"%{user_name}%")).all()
        if not users:
            return f"No previous interactions found for customer: {user_name}"

        history = []
        for user in users:
            tickets = session.query(Ticket).filter_by(user_id=user.user_id).all()
            for ticket in tickets:
                metadata = session.query(TicketMetadata).filter_by(ticket_id=ticket.ticket_id).first()
                messages = session.query(TicketMessage).filter_by(ticket_id=ticket.ticket_id).all()
                history.append({
                    "ticket_id": ticket.ticket_id,
                    "channel": ticket.channel,
                    "status": metadata.status if metadata else "unknown",
                    "issue_type": metadata.main_issue_type if metadata else None,
                    "created_at": str(ticket.created_at),
                    "messages": [
                        {"role": msg.role.value if msg.role else "unknown", "content": msg.content[:100]}
                        for msg in messages[:5]
                    ]
                })

        if not history:
            return f"No ticket history found for customer: {user_name}"

        return f"Customer history for '{user_name}':\n{json.dumps(history, indent=2)}"
    finally:
        session.close()


@tool
def store_customer_preference(user_name: str, preference_key: str, preference_value: str) -> str:
    """Store a customer preference or note for long-term memory. This helps personalize future interactions."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import TicketMessage, User, Ticket, RoleEnum

    session = _get_udahub_session()
    try:
        users = session.query(User).filter(User.user_name.ilike(f"%{user_name}%")).all()
        if not users:
            return f"Cannot store preference: user '{user_name}' not found"

        user = users[0]
        ticket = session.query(Ticket).filter_by(user_id=user.user_id).order_by(Ticket.created_at.desc()).first()

        if ticket:
            memory_content = f"[MEMORY] {preference_key}: {preference_value}"
            message = TicketMessage(
                message_id=str(uuid.uuid4()),
                ticket_id=ticket.ticket_id,
                role=RoleEnum.system,
                content=memory_content,
            )
            session.add(message)
            session.commit()
            return f"Stored preference for {user_name}: {preference_key} = {preference_value}"
        else:
            return f"No ticket context found for user '{user_name}'. Preference noted but not persisted."
    except Exception as e:
        session.rollback()
        return f"Error storing preference: {str(e)}"
    finally:
        session.close()


MEMORY_SYSTEM_PROMPT = """You are a Memory Agent for CultPass customer support.

Your job is to manage long-term memory for personalized customer support.

**Responsibilities:**
1. Retrieve past interactions when a returning customer contacts support
2. Store important preferences and notes about customers
3. Provide historical context to help other agents make better decisions

**When to retrieve history:**
- When a customer name or email is mentioned
- When the conversation suggests this is a returning customer
- When context from past interactions would help resolution

**When to store preferences:**
- Customer explicitly states a preference
- A resolution pattern is established
- Important account notes should be recorded

**Response Format:**
MEMORY:
Context: [Relevant historical context if found]
Preferences: [Known customer preferences]
Recommendation: [How this context should influence the response]
"""


def resolve_customer_context(message: str, known_users: list[dict] | None = None) -> dict:
    """Extract identity hints and retrieve relevant long-term context."""
    known_users = known_users or []
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", message or "")
    email = email_match.group(0) if email_match else None
    name_fragment = _extract_name_fragment(message)

    matched_user = None
    if email:
        for user in known_users:
            if user.get("email", "").lower() == email.lower():
                matched_user = user
                break
    elif name_fragment:
        lowered_name = name_fragment.lower()
        for user in known_users:
            if lowered_name in user.get("name", "").lower() or lowered_name in user.get("full_name", "").lower():
                matched_user = user
                break

    history = None
    if matched_user:
        lookup_name = matched_user.get("name") or matched_user.get("full_name")
        history = retrieve_customer_history.invoke({"user_name": lookup_name})

    preference_signal = None
    preference_match = re.search(r"prefer\s+([a-z ]+)", message or "", re.IGNORECASE)
    if matched_user and preference_match:
        preference_signal = store_customer_preference.invoke({
            "user_name": matched_user.get("name") or matched_user.get("full_name"),
            "preference_key": "support_preference",
            "preference_value": preference_match.group(1).strip(),
        })

    return {
        "matched_user": matched_user,
        "history": history,
        "preference_signal": preference_signal,
    }


memory_agent = {
    "name": "memory",
    "system_prompt": MEMORY_SYSTEM_PROMPT,
    "run": resolve_customer_context,
}
