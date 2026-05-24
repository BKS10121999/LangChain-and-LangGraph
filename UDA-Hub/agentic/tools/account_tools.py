"""Account management tools for CultPass database interactions."""
import json
import os
from langchain_core.tools import tool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys

# Database path relative to project root
CULTPASS_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "external", "cultpass.db")


def _get_cultpass_session():
    """Get a database session for CultPass."""
    engine = create_engine(f"sqlite:///{CULTPASS_DB_PATH}", echo=False)
    Session = sessionmaker(bind=engine)
    return Session()


def lookup_user_record(email: str | None = None, user_id: str | None = None) -> dict | None:
    """Return a structured user record for internal workflow use."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.cultpass import User

    session = _get_cultpass_session()
    try:
        query = session.query(User)
        user = None
        if email:
            user = query.filter_by(email=email).first()
        elif user_id:
            user = query.filter_by(user_id=user_id).first()

        if not user:
            return None

        subscription = user.subscription
        return {
            "user_id": user.user_id,
            "full_name": user.full_name,
            "email": user.email,
            "is_blocked": user.is_blocked,
            "created_at": str(user.created_at),
            "subscription": {
                "subscription_id": subscription.subscription_id,
                "status": subscription.status,
                "tier": subscription.tier,
                "monthly_quota": subscription.monthly_quota,
                "started_at": str(subscription.started_at),
                "ended_at": str(subscription.ended_at) if subscription.ended_at else None,
            } if subscription else None,
        }
    finally:
        session.close()


@tool
def lookup_user(email: str) -> str:
    """Look up a CultPass user by their email address. Returns user details including name, email, blocked status, and subscription info."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.cultpass import User, Subscription

    result = lookup_user_record(email=email)
    if not result:
        return f"No user found with email: {email}"
    return json.dumps(result, indent=2)


@tool
def get_subscription_details(user_id: str) -> str:
    """Get subscription details for a CultPass user by their user ID. Returns tier, status, quota, and dates."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.cultpass import Subscription

    session = _get_cultpass_session()
    try:
        subscription = session.query(Subscription).filter_by(user_id=user_id).first()
        if not subscription:
            return f"No subscription found for user_id: {user_id}"

        return json.dumps({
            "subscription_id": subscription.subscription_id,
            "user_id": subscription.user_id,
            "status": subscription.status,
            "tier": subscription.tier,
            "monthly_quota": subscription.monthly_quota,
            "started_at": str(subscription.started_at),
            "ended_at": str(subscription.ended_at) if subscription.ended_at else None,
        }, indent=2)
    finally:
        session.close()


@tool
def check_user_reservations(user_id: str) -> str:
    """Check all reservations for a CultPass user. Returns list of reservations with experience details and status."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data.models.cultpass import Reservation, Experience

    session = _get_cultpass_session()
    try:
        reservations = session.query(Reservation).filter_by(user_id=user_id).all()
        if not reservations:
            return f"No reservations found for user_id: {user_id}"

        results = []
        for res in reservations:
            experience = session.query(Experience).filter_by(experience_id=res.experience_id).first()
            results.append({
                "reservation_id": res.reservation_id,
                "status": res.status,
                "experience": {
                    "title": experience.title if experience else "Unknown",
                    "location": experience.location if experience else "Unknown",
                    "when": str(experience.when) if experience else "Unknown",
                    "is_premium": experience.is_premium if experience else False,
                } if experience else None,
                "created_at": str(res.created_at),
            })

        return json.dumps(results, indent=2)
    finally:
        session.close()
