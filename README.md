# Multi-Agent Scheduling Assistant

A LangGraph-orchestrated scheduling assistant with a **Triage Agent** and a
**Booking Specialist**, mocked-but-functional calendar tools, and a
Streamlit chat UI whose conversation history survives page refreshes.

## 🏗️ Architecture

```
                     ┌──────────────┐
   user message ───► │ Triage Agent │
                     └──────┬───────┘
                            │
              ┌─────────────┴─────────────┐
        general query                booking intent
              │                             │
              ▼                             ▼
      answered directly           ┌────────────────────┐
             (END)                │ Booking Specialist  │
                                   │  - normalizes dates │
                                   │  - check_availability│
                                   │  - reserve_slot     │
                                   │  - negotiates on    │
                                   │    conflicts        │
                                   │  - send_notification│
                                   └──────────┬──────────┘
                                              ▼
                                            (END)
```

- **`tools.py`** — three mocked-but-functional tools:
  - `check_availability(date)` — reads a local SQLite "calendar".
  - `reserve_slot(date, time, email)` — writes to SQLite; returns
    `alternative_slots` if the requested slot is taken (used to drive
    negotiation).
  - `send_booking_notification(email, details)` — POSTs to a webhook URL
    (e.g. [webhook.site](https://webhook.site)) if `WEBHOOK_URL` is set,
    otherwise simulates success.
- **`graph.py`** — the LangGraph `StateGraph`:
  - `triage_node` classifies intent and answers general questions directly.
  - `booking_node` runs an agentic tool-use loop: it resolves relative
    dates (e.g. "tomorrow") against today's date *before* calling any
    tool, calls tools as needed, and if `reserve_slot` fails, surfaces the
    alternative slots and asks the user to pick one instead of failing
    silently.
  - Conversation state is checkpointed with `SqliteSaver` to
    `checkpoints.sqlite`, keyed by `thread_id`.
- **`app.py`** — Streamlit UI. The thread id lives in the URL query
  string, so refreshing the browser resumes the same conversation by
  reloading it from the checkpoint DB.

## 🚀 Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."
# optional — get a free URL at https://webhook.site to see real notification payloads
export WEBHOOK_URL="https://webhook.site/your-unique-url"
streamlit run app.py
```

Try it:
- *"What are your business hours?"* → handled by the Triage Agent directly.
- *"Book me a slot tomorrow at 9am, email me at me@example.com"* → routed
  to the Booking Specialist, which resolves "tomorrow" to a real date and
  books it.
- Ask for the same slot again with a different email → the specialist
  detects the conflict and proposes alternatives instead of failing.

## ☁️ Deploying (free tier)

**Render**
1. New **Web Service** from this repo.
2. Build command: `pip install -r requirements.txt`
3. Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add `ANTHROPIC_API_KEY` (and optionally `WEBHOOK_URL`) as environment variables.

**Hugging Face Spaces**
1. New Space → SDK: Streamlit → upload these files.
2. Add secrets under **Settings → Repository secrets**.

> Note: on ephemeral free-tier hosts, the SQLite files (`bookings.db`,
> `checkpoints.sqlite`) reset on redeploy. For durable persistence beyond
> that, swap in a hosted Postgres/Redis checkpointer — LangGraph supports
> drop-in alternatives to `SqliteSaver`.

## 🔒 Notes

- No API keys are committed; both `ANTHROPIC_API_KEY` and `WEBHOOK_URL`
  are read from the environment / hosting secrets.
- Business hours and slot granularity (hourly, 9-16 minus lunch) are
  defined in `tools.py` and easy to adjust.
