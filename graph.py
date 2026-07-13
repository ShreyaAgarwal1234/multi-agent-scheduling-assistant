"""
graph.py
--------
LangGraph state-machine that orchestrates two agents:

  Triage Agent      -> classifies the user's message. General questions are
                        answered directly. Scheduling intent routes to the
                        Booking Specialist.
  Booking Specialist -> normalizes relative dates ("tomorrow" -> YYYY-MM-DD),
                        drives the check_availability / reserve_slot /
                        send_booking_notification tool loop, and negotiates
                        alternative slots if a booking fails.

Conversation state is checkpointed with LangGraph's SqliteSaver so a
thread's history survives page refreshes.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from google import genai
from google.genai import types
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import TOOL_FUNCTIONS

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")

# Holds open sqlite3 connections so they are never garbage-collected while
# the app is running (see build_graph() below).
_OPEN_CONNECTIONS: list = []


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # "general" | "booking"


def _client() -> genai.Client:
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def _to_gemini_contents(messages: list) -> list[dict]:
    """Convert LangGraph message objects into Gemini 'contents' dicts."""
    out = []
    for m in messages:
        role = "user" if m.type in ("human",) else "model"
        out.append({"role": role, "parts": [{"text": m.content}]})
    return out


# --------------------------------------------------------------------------
# Triage Agent
# --------------------------------------------------------------------------
TRIAGE_SYSTEM = """You are a triage classifier for a scheduling assistant.
Decide if the user's LATEST message expresses intent to schedule, check,
reschedule, cancel, or book an appointment. Respond with EXACTLY one word:
"booking" if there is any scheduling intent, or "general" otherwise.
No punctuation, no explanation."""


def triage_node(state: AgentState) -> dict:
    client = _client()
    contents = _to_gemini_contents(state["messages"])

    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=TRIAGE_SYSTEM,
            max_output_tokens=40,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    decision = (resp.text or "").strip().lower()
    route = "booking" if "booking" in decision else "general"

    if route == "general":
        # Answer directly for general queries.
        reply = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are a friendly scheduling assistant's front desk. Answer "
                    "general questions helpfully and briefly. If relevant, mention "
                    "that you can also help book, check, or reschedule appointments."
                ),
                max_output_tokens=400,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        text = reply.text or ""
        return {"messages": [{"role": "assistant", "content": text}], "route": "general"}

    return {"route": "booking"}


# --------------------------------------------------------------------------
# Booking Specialist
# --------------------------------------------------------------------------
def _booking_system_prompt() -> str:
    today = datetime.now().strftime("%Y-%m-%d (%A)")
    return f"""You are the Booking Specialist for a scheduling assistant.
Today's date is {today}.

Rules:
1. ALWAYS resolve relative dates ("tomorrow", "next Monday", "this Friday")
   to an actual YYYY-MM-DD string yourself before calling any tool. Never
   pass a relative date string into a tool.
2. Business hours are 09:00-16:00 in 1-hour slots (09:00, 10:00, 11:00,
   13:00, 14:00, 15:00, 16:00). No slots at 12:00 (lunch).
3. Use check_availability before reserve_slot when the user hasn't
   specified an exact time, or when you're unsure a slot is free.
4. If reserve_slot fails because the slot is taken, DO NOT fail silently.
   Look at the alternative_slots returned and proactively suggest 2-3
   nearby alternatives to the user, and ask them to pick one.
5. If you're missing the customer's email, date, or time, ask for exactly
   the missing piece(s) — don't ask for information you already have.
6. Once a slot is successfully reserved, ALWAYS call
   send_booking_notification to confirm it, then tell the user it's booked
   with a short friendly confirmation summarizing date, time, and email.
7. Be concise and conversational."""


def booking_node(state: AgentState) -> dict:
    client = _client()
    contents = _to_gemini_contents(state["messages"])
    system = _booking_system_prompt()

    # Gemini's automatic function calling: pass the plain Python functions
    # directly as tools. The SDK reads each function's type hints and
    # docstring to build the schema, calls whichever tools the model
    # decides it needs (in a loop internally), and returns the final
    # text response once the model is done calling tools.
    resp = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            tools=list(TOOL_FUNCTIONS.values()),
        ),
    )

    final_text = resp.text or "Let me know a date, time, and email and I'll get that booked."
    return {"messages": [{"role": "assistant", "content": final_text}], "route": "booking"}


def _route_after_triage(state: AgentState) -> Literal["booking_specialist", "__end__"]:
    return "booking_specialist" if state["route"] == "booking" else END


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("triage", triage_node)
    graph.add_node("booking_specialist", booking_node)
    graph.set_entry_point("triage")
    graph.add_conditional_edges("triage", _route_after_triage, {
        "booking_specialist": "booking_specialist",
        END: END,
    })
    graph.add_edge("booking_specialist", END)

    # Open the sqlite connection directly (instead of via the
    # SqliteSaver.from_conn_string(...) context manager) and keep a
    # reference to it in a module-level list. The context-manager form
    # can get garbage-collected once this function returns, which closes
    # the underlying connection out from under the checkpointer and
    # causes "Cannot operate on a closed database" errors. check_same_thread=False
    # is needed because Streamlit may call into the graph from a
    # different thread than the one that opened the connection.
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    _OPEN_CONNECTIONS.append(conn)  # prevent garbage collection
    checkpointer = SqliteSaver(conn)
    return graph.compile(checkpointer=checkpointer)