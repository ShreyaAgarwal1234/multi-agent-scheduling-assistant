"""
tools.py
--------
Mocked-but-functional tools for the scheduling assistant:
  - check_availability(date): looks up a date in an in-memory "calendar"
  - reserve_slot(date, time, email): books a slot in a local SQLite DB
  - send_booking_notification(email, details): fires a mock webhook

None of these need real external services to run — reserve_slot persists
to a local SQLite file, and send_booking_notification posts to
webhook.site if WEBHOOK_URL is set, otherwise just logs the payload.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime

import requests

DB_PATH = os.path.join(os.path.dirname(__file__), "bookings.db")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")  # e.g. a webhook.site URL

# A small in-memory "calendar" of already-taken slots, seeded with a few
# busy times so the negotiation flow has something real to react to.
_TAKEN_SLOTS: dict[str, set[str]] = {}


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            email TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


_init_db()

BUSINESS_HOURS = ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]


def check_availability(date: str) -> dict:
    """Check which slots are free on a given YYYY-MM-DD date.

    Returns: {"date": ..., "available_slots": [...], "taken_slots": [...]}
    """
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format '{date}', expected YYYY-MM-DD"}

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT time FROM bookings WHERE date = ?", (date,)).fetchall()
    conn.close()

    taken = {r[0] for r in rows} | _TAKEN_SLOTS.get(date, set())
    available = [t for t in BUSINESS_HOURS if t not in taken]

    return {
        "date": date,
        "available_slots": available,
        "taken_slots": sorted(taken),
    }


def reserve_slot(date: str, time: str, email: str) -> dict:
    """Reserve a slot. Fails if the slot is already taken (used to trigger
    the negotiation flow in the Booking Specialist agent)."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"success": False, "error": f"Invalid date format '{date}'"}

    if time not in BUSINESS_HOURS:
        return {
            "success": False,
            "error": f"'{time}' is outside business hours. Valid slots: {BUSINESS_HOURS}",
        }

    availability = check_availability(date)
    if time in availability["taken_slots"]:
        return {
            "success": False,
            "error": f"Slot {date} {time} is already taken.",
            "alternative_slots": availability["available_slots"],
        }

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO bookings (date, time, email, created_at) VALUES (?, ?, ?, ?)",
        (date, time, email, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()

    return {"success": True, "date": date, "time": time, "email": email}


def send_booking_notification(email: str, details: str) -> dict:
    """Simulate sending an email/WhatsApp confirmation via a mock webhook.
    Posts to WEBHOOK_URL if configured (e.g. a webhook.site test URL),
    otherwise just returns a simulated success payload."""
    payload = {
        "to": email,
        "message": details,
        "sent_at": datetime.utcnow().isoformat(),
    }

    if WEBHOOK_URL:
        try:
            resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
            return {"success": True, "status_code": resp.status_code, "payload": payload}
        except requests.RequestException as e:
            return {"success": False, "error": str(e), "payload": payload}

    # No webhook configured — simulate success so the flow still works.
    return {"success": True, "simulated": True, "payload": payload}


TOOL_SCHEMAS = [
    {
        "name": "check_availability",
        "description": "Check available appointment slots for a given date (YYYY-MM-DD).",
        "input_schema": {
            "type": "object",
            "properties": {"date": {"type": "string", "description": "Date in YYYY-MM-DD format"}},
            "required": ["date"],
        },
    },
    {
        "name": "reserve_slot",
        "description": "Reserve an appointment slot for a given date, time, and customer email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Time in HH:MM 24-hour format"},
                "email": {"type": "string", "description": "Customer email address"},
            },
            "required": ["date", "time", "email"],
        },
    },
    {
        "name": "send_booking_notification",
        "description": "Send a booking confirmation notification to the customer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {"type": "string"},
                "details": {"type": "string", "description": "Human-readable booking details"},
            },
            "required": ["email", "details"],
        },
    },
]

TOOL_FUNCTIONS = {
    "check_availability": check_availability,
    "reserve_slot": reserve_slot,
    "send_booking_notification": send_booking_notification,
}
