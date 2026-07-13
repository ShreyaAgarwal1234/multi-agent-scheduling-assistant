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

import json
import os
from datetime import datetime
from typing import Annotated, Literal, TypedDict

import sqlite3

import anthropic
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

MODEL_NAME = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CHECKPOINT_DB = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")

# Holds open sqlite3 connections so they are never garbage-collected while
# the app is running (see build_graph() below).
_OPEN_CONNECTIONS: list = []


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str  # "general" | "booking"


def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


def _to_anthropic_messages(messages: list) -> list[dict]:
    """Convert LangGraph message objects into plain Anthropic API dicts."""
    out = []
    for m in messages:
        role = "user" if m.type in ("human",) else "assistant"
        out.append({"role": role, "content": m.content})
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
    history = _to_anthropic_messages(state["messages"])

    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=10,
        system=TRIAGE_SYSTEM,
        messages=history,
    )
    decision = resp.content[0].text.strip().lower()
    route = "booking" if "booking" in decision else "general"

    if route == "general":
        # Answer directly for general queries.
        reply = client.messages.create(
            model=MODEL_NAME,
            max_tokens=400,
            system=(
                "You are a friendly scheduling assistant's front desk. Answer "
                "general questions helpfully and briefly. If relevant, mention "
                "that you can also help book, check, or reschedule appointments."
            ),
            messages=history,
        )
        text = "".join(b.text for b in reply.content if b.type == "text")
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
    history = _to_anthropic_messages(state["messages"])
    system = _booking_system_prompt()

    new_messages = []
    # Agentic tool-use loop (capped to avoid runaway loops).
    for _ in range(6):
        resp = client.messages.create(
            model=MODEL_NAME,
            max_tokens=800,
            system=system,
            tools=TOOL_SCHEMAS,
            messages=history + new_messages,
        )

        assistant_content = resp.content
        new_messages.append({"role": "assistant", "content": assistant_content})

        if resp.stop_reason != "tool_use":
            break

        tool_results = []
        for block in assistant_content:
            if block.type != "tool_use":
                continue
            fn = TOOL_FUNCTIONS.get(block.name)
            result = fn(**block.input) if fn else {"error": f"Unknown tool {block.name}"}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                }
            )
        new_messages.append({"role": "user", "content": tool_results})

    final_text = ""
    if new_messages and isinstance(new_messages[-1]["content"], list):
        for block in new_messages[-1]["content"]:
            if getattr(block, "type", None) == "text":
                final_text += block.text
    if not final_text:
        final_text = "Let me know a date, time, and email and I'll get that booked."

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