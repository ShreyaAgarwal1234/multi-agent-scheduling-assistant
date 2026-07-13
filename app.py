"""
app.py — Multi-Agent Scheduling Assistant
Streamlit front end for the LangGraph Triage Agent + Booking Specialist
workflow. Thread state is checkpointed to SQLite and the thread id is
kept in the URL query params, so conversation history survives page
refreshes.
"""

import os
import uuid

import streamlit as st

from graph import build_graph

st.set_page_config(
    page_title="Scheduling Assistant",
    page_icon="🗓️",
    layout="centered",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}
    .stApp {background: linear-gradient(180deg, #0c1220 0%, #131a2b 100%);}
    .block-container {padding-top: 2rem; max-width: 780px;}
    h1, h2, h3, p, span, label, .stMarkdown {color: #e5e7eb;}

    .sc-header {
        display: flex; align-items: center; gap: 12px;
        padding: 18px 22px; border-radius: 16px;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
        box-shadow: 0 8px 24px rgba(2,132,199,0.35);
        margin-bottom: 22px;
    }
    .sc-header h1 {font-size: 1.35rem; margin: 0; color: white;}
    .sc-header p {font-size: 0.85rem; margin: 0; color: #e0f2fe;}
    .sc-badge {
        margin-left: auto; font-size: 0.7rem; background: #ffffff22;
        color: #e0f2fe; padding: 4px 10px; border-radius: 999px;
        border: 1px solid #ffffff33; white-space: nowrap;
    }

    .route-pill {
        display: inline-block; font-size: 0.68rem; padding: 2px 10px;
        border-radius: 999px; margin-bottom: 6px; font-weight: 600;
        letter-spacing: 0.02em;
    }
    .route-general {background: #6366f122; color: #a5b4fc; border: 1px solid #6366f155;}
    .route-booking {background: #0ea5e922; color: #7dd3fc; border: 1px solid #0ea5e955;}

    section[data-testid="stSidebar"] {background: #0c1220; border-right: 1px solid #1f2937;}
    .sidebar-card {
        background: #131a2b; border-radius: 12px; padding: 14px 16px;
        margin-bottom: 14px; border: 1px solid #1e293b;
    }
    .sidebar-card h4 {margin: 0 0 6px 0; color: #7dd3fc; font-size: 0.85rem;}
    .sidebar-card p {margin: 0; font-size: 0.78rem; color: #94a3b8; line-height: 1.5;}
    .thread-id {font-family: monospace; font-size: 0.7rem; color: #64748b;}

    .stChatInput textarea {
        background: #131a2b !important; color: #e5e7eb !important;
        border-radius: 14px !important; border: 1px solid #1e293b !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="sc-header">
        <div style="font-size: 1.8rem;">🗓️</div>
        <div>
            <h1>Scheduling Assistant</h1>
            <p>Triage Agent + Booking Specialist, powered by LangGraph</p>
        </div>
        <div class="sc-badge">● session persisted</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Thread id — kept in the URL so refreshing the page resumes the same
# conversation (LangGraph's SqliteSaver checkpoints by thread_id).
# --------------------------------------------------------------------------
if "thread_id" not in st.query_params:
    st.query_params["thread_id"] = str(uuid.uuid4())[:8]
thread_id = st.query_params["thread_id"]

with st.sidebar:
    st.markdown("### ⚙️ How it works")
    st.markdown(
        """
        <div class="sidebar-card">
            <h4>🧭 Triage Agent</h4>
            <p>Reads your message and decides: general question, or
            scheduling intent? General questions get answered directly.</p>
        </div>
        <div class="sidebar-card">
            <h4>📋 Booking Specialist</h4>
            <p>Resolves relative dates ("tomorrow" → real date), checks
            availability, reserves slots, and negotiates alternatives if
            your requested time is taken.</p>
        </div>
        <div class="sidebar-card">
            <h4>💾 Persistence</h4>
            <p>This conversation is thread <span class="thread-id">{tid}</span> —
            refresh the page and it'll still be here.</p>
        </div>
        """.format(tid=thread_id),
        unsafe_allow_html=True,
    )
    if st.button("🆕 Start new conversation", use_container_width=True):
        st.query_params["thread_id"] = str(uuid.uuid4())[:8]
        st.rerun()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Set ANTHROPIC_API_KEY to enable the agents.", icon="⚠️")

# --------------------------------------------------------------------------
# Graph (cached across reruns, keyed by thread persistence db)
# --------------------------------------------------------------------------
@st.cache_resource
def get_compiled_graph():
    return build_graph()

app = get_compiled_graph()
config = {"configurable": {"thread_id": thread_id}}

# --------------------------------------------------------------------------
# Render existing history from the checkpoint
# --------------------------------------------------------------------------
try:
    snapshot = app.get_state(config)
    history = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
except Exception:
    history = []

for m in history:
    role = "user" if m.type == "human" else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🗓️"
    with st.chat_message(role, avatar=avatar):
        st.markdown(m.content)

# --------------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------------
user_input = st.chat_input("Try: \"Book me a slot tomorrow at 9am\"")

if user_input:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🗓️"):
        with st.spinner("Routing to the right agent…"):
            if not os.environ.get("ANTHROPIC_API_KEY"):
                st.error("No ANTHROPIC_API_KEY set in the environment.")
            else:
                try:
                    result = app.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=config,
                    )
                    route = result.get("route", "general")
                    pill_class = "route-booking" if route == "booking" else "route-general"
                    pill_label = "📋 Booking Specialist" if route == "booking" else "🧭 Triage Agent"
                    st.markdown(f'<span class="route-pill {pill_class}">{pill_label}</span>', unsafe_allow_html=True)

                    last_msg = result["messages"][-1]
                    st.markdown(last_msg.content)
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
