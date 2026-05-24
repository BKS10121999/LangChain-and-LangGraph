"""Ticket management tools for UDA-Hub database."""
import json
import os
import sys
import uuid
from langchain_core.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Database path
UDAHUB_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "core", "udahub.db")


def _get_udahub_session():
    """Get a database session for UDA-Hub."""
    engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


def get_latest_ticket_for_external_user(external_user_id: str, account_id: str = "cultpass") -> dict | None:
    """Return the latest ticket and metadata for an external user."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import User, Ticket, TicketMetadata

    session = _get_udahub_session()
    try:
        user = session.query(User).filter_by(account_id=account_id, external_user_id=external_user_id).first()
        if not user:
            return None

        ticket = session.query(Ticket).filter_by(user_id=user.user_id).order_by(Ticket.created_at.desc()).first()
        if not ticket:
            return None

        metadata = session.query(TicketMetadata).filter_by(ticket_id=ticket.ticket_id).first()
        return {
            "ticket_id": ticket.ticket_id,
            "user_id": user.user_id,
            "user_name": user.user_name,
            "status": metadata.status if metadata else None,
            "issue_type": metadata.main_issue_type if metadata else None,
            "tags": metadata.tags if metadata else None,
            "channel": ticket.channel,
            "created_at": str(ticket.created_at),
        }
    finally:
        session.close()


def fetch_ticket_messages(ticket_id: str) -> list[dict]:
    """Return ordered ticket messages for internal workflow memory use."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import TicketMessage

    session = _get_udahub_session()
    try:
        messages = session.query(TicketMessage).filter_by(ticket_id=ticket_id).order_by(TicketMessage.created_at.asc()).all()
        return [
            {
                "message_id": message.message_id,
                "role": message.role.value,
                "content": message.content,
                "created_at": str(message.created_at),
            }
            for message in messages
        ]
    finally:
        session.close()


@tool
def update_ticket_status(ticket_id: str, new_status: str, issue_type: str = None) -> str:
    """Update the status and optional issue type of a support ticket. Valid statuses: open, in_progress, resolved, escalated, closed."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import TicketMetadata

    valid_statuses = ["open", "in_progress", "resolved", "escalated", "closed"]
    if new_status not in valid_statuses:
        return f"Invalid status '{new_status}'. Valid statuses: {valid_statuses}"

    session = _get_udahub_session()
    try:
        metadata = session.query(TicketMetadata).filter_by(ticket_id=ticket_id).first()
        if not metadata:
            return f"No ticket found with id: {ticket_id}"

        old_status = metadata.status
        metadata.status = new_status
        if issue_type:
            metadata.main_issue_type = issue_type
        session.commit()

        return f"Ticket {ticket_id} status updated from '{old_status}' to '{new_status}'" + (f", issue type set to '{issue_type}'" if issue_type else "")
    except Exception as e:
        session.rollback()
        return f"Error updating ticket: {str(e)}"
    finally:
        session.close()


@tool
def log_ticket_message(ticket_id: str, role: str, content: str) -> str:
    """Log a message to a ticket's conversation history. Role must be one of: user, agent, ai, system."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.udahub import TicketMessage, RoleEnum

    valid_roles = ["user", "agent", "ai", "system"]
    if role not in valid_roles:
        return f"Invalid role '{role}'. Valid roles: {valid_roles}"

    session = _get_udahub_session()
    try:
        message = TicketMessage(
            message_id=str(uuid.uuid4()),
            ticket_id=ticket_id,
            role=role,
            content=content,
        )
        session.add(message)
        session.commit()
        return f"Message logged to ticket {ticket_id} (role: {role})"
    except Exception as e:
        session.rollback()
        return f"Error logging message: {str(e)}"
    finally:
        session.close()


@tool
def get_ticket_history(ticket_id: str) -> str:
    """Get the full conversation history for a ticket. Use this for personalized follow-ups and audits."""
    messages = fetch_ticket_messages(ticket_id)
    if not messages:
        return f"No ticket history found for ticket_id: {ticket_id}"
    return json.dumps(messages, indent=2)
