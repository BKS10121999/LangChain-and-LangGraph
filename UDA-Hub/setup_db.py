"""Database setup script - Run this to initialize both databases.

Usage: python setup_db.py
"""
import sys
import os
import json
import uuid
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine
from utils import get_session
from data.models import cultpass, udahub


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DB = os.path.join(PROJECT_ROOT, "data", "external", "cultpass.db")
CORE_DB = os.path.join(PROJECT_ROOT, "data", "core", "udahub.db")
ARTICLES_PATH = os.path.join(PROJECT_ROOT, "data", "external", "cultpass_articles.jsonl")
USERS_PATH = os.path.join(PROJECT_ROOT, "data", "external", "cultpass_users.jsonl")
EXPERIENCES_PATH = os.path.join(PROJECT_ROOT, "data", "external", "cultpass_experiences.jsonl")

RESERVATION_STATUSES = ["reserved", "attended", "cancelled"]
SUBSCRIPTION_STATUSES = ["active", "cancelled"]
SUBSCRIPTION_TIERS = ["basic", "premium"]


def _load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _seeded_random():
    return random.Random(42)


def _reset_database(db_path, metadata):
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print(f"  Removed existing {db_path}")
            engine = create_engine(f"sqlite:///{db_path}", echo=False)
            metadata.create_all(engine)
            return engine
        except PermissionError:
            print(f"  Could not remove {db_path}; resetting schema in place")

    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    metadata.drop_all(engine)
    metadata.create_all(engine)
    return engine


def _sample_ticket_templates():
    return [
        {
            "content": "I can't log in to my CultPass account.",
            "channel": "chat",
            "status": "open",
            "issue_type": "technical",
            "tags": "login, access",
            "history": [
                ("user", "I can't log in to my CultPass account."),
                ("system", "[MEMORY] prefers concise troubleshooting steps"),
            ],
        },
        {
            "content": "Can I change from Basic to Premium before my trip next week?",
            "channel": "chat",
            "status": "resolved",
            "issue_type": "subscription",
            "tags": "upgrade, premium, plan",
            "history": [
                ("user", "Can I change from Basic to Premium before my trip next week?"),
                ("ai", "Yes. You can upgrade anytime from My Account > Manage Plan > Upgrade."),
                ("system", "[MEMORY] interested in premium events during travel"),
            ],
        },
        {
            "content": "My payment failed and now my account is blocked.",
            "channel": "email",
            "status": "escalated",
            "issue_type": "billing",
            "tags": "payment, blocked, billing",
            "history": [
                ("user", "My payment failed and now my account is blocked."),
                ("agent", "We need a billing specialist to review the failed charges and unblock request."),
            ],
        },
        {
            "content": "I need to know whether my reservation for the gallery walk is still active.",
            "channel": "chat",
            "status": "resolved",
            "issue_type": "reservation",
            "tags": "reservation, status, qr",
            "history": [
                ("user", "I need to know whether my reservation for the gallery walk is still active."),
                ("ai", "You can review active reservations in My Reservations, where the QR code and venue details are listed."),
            ],
        },
        {
            "content": "Please delete my account and send me a copy of my data.",
            "channel": "web",
            "status": "in_progress",
            "issue_type": "account",
            "tags": "privacy, deletion, data",
            "history": [
                ("user", "Please delete my account and send me a copy of my data."),
                ("agent", "Identity verification is required before we can process deletion and export requests."),
            ],
        },
        {
            "content": "The app keeps crashing when I open my ticket QR code.",
            "channel": "chat",
            "status": "open",
            "issue_type": "technical",
            "tags": "app, crash, qr",
            "history": [
                ("user", "The app keeps crashing when I open my ticket QR code."),
            ],
        },
    ]


def setup_external_db():
    """Set up the CultPass external database."""
    print("Setting up CultPass external database...")
    cultpass_db = EXTERNAL_DB
    rng = _seeded_random()

    engine = _reset_database(cultpass_db, cultpass.Base.metadata)

    # Load experiences
    experience_data = []
    experience_data = _load_jsonl(EXPERIENCES_PATH)

    with get_session(engine) as session:
        for idx, exp in enumerate(experience_data):
            session.add(cultpass.Experience(
                experience_id=str(uuid.uuid4())[:6],
                title=exp["title"],
                description=exp["description"],
                location=exp["location"],
                when=datetime.now() + timedelta(days=idx + 1),
                slots_available=rng.randint(1, 30),
                is_premium=(idx % 2 == 0)
            ))

    # Load users
    cultpass_users = _load_jsonl(USERS_PATH)

    with get_session(engine) as session:
        for user_info in cultpass_users:
            session.add(cultpass.User(
                user_id=user_info["id"],
                full_name=user_info["name"],
                email=user_info["email"],
                is_blocked=user_info["is_blocked"],
                created_at=datetime.now()
            ))

    # Subscriptions
    with get_session(engine) as session:
        for user_info in cultpass_users:
            session.add(cultpass.Subscription(
                subscription_id=str(uuid.uuid4())[:6],
                user_id=user_info["id"],
                status=SUBSCRIPTION_STATUSES[len(user_info["id"]) % len(SUBSCRIPTION_STATUSES)],
                tier=SUBSCRIPTION_TIERS[len(user_info["email"]) % len(SUBSCRIPTION_TIERS)],
                monthly_quota=8 if len(user_info["name"]) % 2 == 0 else 4,
                started_at=datetime.now()
            ))

    # Reservations
    with get_session(engine) as session:
        experience_ids = [e.experience_id for e in session.query(cultpass.Experience).all()]
        for user_index, user_info in enumerate(cultpass_users):
            reservation_count = 2 + (user_index % 2)
            for offset in range(reservation_count):
                session.add(cultpass.Reservation(
                    reservation_id=str(uuid.uuid4())[:6],
                    user_id=user_info["id"],
                    experience_id=experience_ids[(user_index + offset) % len(experience_ids)],
                    status=RESERVATION_STATUSES[(user_index + offset) % len(RESERVATION_STATUSES)],
                ))

    print(f"  Users: {len(cultpass_users)}")
    print(f"  Experiences: {len(experience_data)}")
    print(f"  Reservations: {sum(2 + (idx % 2) for idx, _ in enumerate(cultpass_users))}")
    print("  Done!")


def setup_core_db():
    """Set up the UDA-Hub core database."""
    print("Setting up UDA-Hub core database...")
    udahub_db = CORE_DB

    engine = _reset_database(udahub_db, udahub.Base.metadata)

    account_id = "cultpass"
    with get_session(engine) as session:
        session.add(udahub.Account(account_id=account_id, account_name="CultPass Card"))

    # Load knowledge base
    cultpass_articles = _load_jsonl(ARTICLES_PATH)

    print(f"  Articles loaded: {len(cultpass_articles)}")
    assert len(cultpass_articles) >= 14, f"Need at least 14, got {len(cultpass_articles)}"

    with get_session(engine) as session:
        for article in cultpass_articles:
            session.add(udahub.Knowledge(
                article_id=str(uuid.uuid4()),
                account_id=account_id,
                title=article["title"],
                content=article["content"],
                tags=article["tags"]
            ))

    # Create mirrored users and representative tickets for memory-driven support.
    cultpass_users = _load_jsonl(USERS_PATH)
    sample_tickets = _sample_ticket_templates()

    with get_session(engine) as session:
        internal_users = {}
        for user_info in cultpass_users:
            user = udahub.User(
                user_id=str(uuid.uuid4()),
                account_id=account_id,
                external_user_id=user_info["id"],
                user_name=user_info["name"],
            )
            session.add(user)
            internal_users[user_info["id"]] = user

        session.flush()

        tickets_created = 0
        for user_index, user_info in enumerate(cultpass_users):
            internal_user = internal_users[user_info["id"]]
            template = sample_tickets[user_index % len(sample_tickets)]
            ticket = udahub.Ticket(
                ticket_id=str(uuid.uuid4()),
                account_id=account_id,
                user_id=internal_user.user_id,
                channel=template["channel"],
            )
            session.add(ticket)
            session.flush()

            metadata = udahub.TicketMetadata(
                ticket_id=ticket.ticket_id,
                status=template["status"],
                main_issue_type=template["issue_type"],
                tags=template["tags"],
            )
            session.add(metadata)

            for role, content in template["history"]:
                session.add(udahub.TicketMessage(
                    message_id=str(uuid.uuid4()),
                    ticket_id=ticket.ticket_id,
                    role=role,
                    content=content,
                ))

            tickets_created += 1

    print(f"  Mirrored users: {len(cultpass_users)}")
    print(f"  Knowledge articles: {len(cultpass_articles)}")
    print(f"  Tickets seeded: {tickets_created}")

    print("  Done!")


if __name__ == "__main__":
    setup_external_db()
    setup_core_db()
    print("\nAll databases initialized successfully!")
