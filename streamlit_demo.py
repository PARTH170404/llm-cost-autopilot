"""
demo/streamlit_demo.py
Live demo for Streamlit Cloud — uses Groq API (free) instead of local Ollama.
Groq provides llama3, mistral, and gemma free with generous rate limits.
"""

import time
import uuid
import sqlite3
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="🚀",
    layout="wide",
)

# ── Groq client (free API) ───────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Model mapping: tier → Groq model name
TIER_MODEL_MAP = {
    "simple":  {"groq": "gemma2-9b-it",       "display": "Gemma 2 9B",   "cost_per_1k": 0.0},
    "medium":  {"groq": "mistral-saba-24b",    "display": "Mistral 24B",  "cost_per_1k": 0.0},
    "complex": {"groq": "llama-3.3-70b-versatile", "display": "Llama 3.3 70B", "cost_per_1k": 0.0},
}
BASELINE_COST_PER_1K = 0.002  # GPT-4 equivalent

# Simple keyword-based classifier (no sklearn needed for demo)
COMPLEX_KW = {"design","architect","distributed","scalable","system","fault-tolerant",
               "implement","real-time","billion","million","global","pipeline","migration",
               "platform","infrastructure","multi-region","orchestration","canary"}
MEDIUM_KW  = {"explain","write","difference","how does","function","class","algorithm",
               "pattern","query","script","compare","analyze","describe","build","create"}

def classify(prompt: str) -> str:
    lower = prompt.lower()
    words = set(lower.split())
    if any(kw in lower for kw in COMPLEX_KW) or len(prompt.split()) > 40:
        return "complex"
    if any(kw in lower for kw in MEDIUM_KW) or len(prompt.split()) > 15:
        return "medium"
    return "simple"

# ── Tiny in-memory SQLite for session ────────────────────────────────────────
@st.cache_resource
def get_db():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS requests (
        id TEXT, timestamp TEXT, prompt TEXT, tier TEXT,
        model TEXT, tokens INTEGER, latency_ms REAL,
        cost REAL, savings REAL, content TEXT
    )""")
    conn.commit()
    return conn

def log_req(conn, **kwargs):
    conn.execute(
        "INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?)",
        (str(uuid.uuid4()), datetime.utcnow().isoformat(),
         kwargs["prompt"], kwargs["tier"], kwargs["model"],
         kwargs["tokens"], kwargs["latency_ms"],
         kwargs["cost"], kwargs["savings"], kwargs["content"])
    )
    conn.commit()

def get_stats(conn):
    cur = conn.execute(
        "SELECT COUNT(*),SUM(tokens),SUM(cost),SUM(savings),AVG(latency_ms) FROM requests"
    )
    row = cur.fetchone()
    dist = conn.execute(
        "SELECT model, COUNT(*) FROM requests GROUP BY model"
    ).fetchall()
    tier_dist = conn.execute(
        "SELECT tier, COUNT(*) FROM requests GROUP BY tier"
    ).fetchall()
    return {
        "total": row[0] or 0, "tokens": int(row[1] or 0),
        "cost": row[2] or 0.0, "savings": row[3] or 0.0,
        "latency": row[4] or 0.0,
        "models": {r[0]: r[1] for r in dist},
        "tiers": {r[0]: r[1] for r in tier_dist},
    }

def get_history(conn, n=30):
    rows = conn.execute(
        "SELECT timestamp,tier,model,tokens,latency_ms,savings FROM requests ORDER BY timestamp DESC LIMIT ?",
        (n,)
    ).fetchall()
    return rows

# ── Call Groq ─────────────────────────────────────────────────────────────────
def call_groq(tier: str, prompt: str, max_tokens: int = 400) -> dict:
    if not GROQ_AVAILABLE or not GROQ_API_KEY:
        # Fallback: simulate response for demo without API key
        time.sleep(0.3)
        content = f"[Demo mode — no API key] This is a simulated response for a {tier} prompt. In production this routes to {TIER_MODEL_MAP[tier]['display']}."
        tokens = len(prompt.split()) + len(content.split())
        return {"content": content, "tokens": tokens, "latency_ms": 300, "error": None}

    client = Groq(api_key=GROQ_API_KEY)
    cfg = TIER_MODEL_MAP[tier]
    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=cfg["groq"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        latency = (time.time() - t0) * 1000
        content = resp.choices[0].message.content
        tokens = resp.usage.total_tokens if resp.usage else len(content.split()) * 2
        return {"content": content, "tokens": tokens, "latency_ms": latency, "error": None}
    except Exception as e:
        return {"content": "", "tokens": 0, "latency_ms": 0, "error": str(e)}

# ── UI ────────────────────────────────────────────────────────────────────────
conn = get_db()

st.title("🚀 LLM Cost Autopilot")
st.caption("Routes every prompt to the cheapest capable model — zero cloud API cost in production (local Ollama)")

if not GROQ_API_KEY:
    st.info("Running in **demo simulation mode** — add a GROQ_API_KEY secret in Streamlit Cloud settings for live AI responses.", icon="ℹ️")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Try it live")
    sample_prompts = {
        "Simple — facts": "What is the capital of Japan?",
        "Medium — coding": "Write a Python function to check if a number is prime.",
        "Complex — system design": "Design a distributed rate limiter that works across 50 servers using Redis.",
    }
    selected = st.selectbox("Load a sample prompt", ["(type your own)"] + list(sample_prompts.keys()))
    user_prompt = st.text_area(
        "Your prompt",
        value=sample_prompts.get(selected, ""),
        height=120,
        placeholder="Ask anything — simple, medium, or complex..."
    )
    max_tokens = st.slider("Max response tokens", 50, 600, 300)
    submit = st.button("Route & Generate", type="primary", use_container_width=True)

# ── Process ───────────────────────────────────────────────────────────────────
if submit and user_prompt.strip():
    tier = classify(user_prompt)
    model_info = TIER_MODEL_MAP[tier]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Detected tier", tier.upper())
    col_b.metric("Routed to", model_info["display"])
    col_c.metric("Baseline (GPT-4)", "→ skipped")

    with st.spinner(f"Calling {model_info['display']}..."):
        result = call_groq(tier, user_prompt, max_tokens)

    if result["error"]:
        st.error(f"Model error: {result['error']}")
    else:
        st.success("Response generated")
        st.markdown("**Response:**")
        st.write(result["content"])

        baseline_cost = (result["tokens"] / 1000) * BASELINE_COST_PER_1K
        actual_cost = 0.0
        savings = baseline_cost

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tokens used", f"{result['tokens']:,}")
        m2.metric("Latency", f"{result['latency_ms']:.0f}ms")
        m3.metric("Actual cost", "$0.0000")
        m4.metric("Est. savings vs GPT-4", f"${savings:.5f}")

        log_req(conn, prompt=user_prompt, tier=tier, model=model_info["display"],
                tokens=result["tokens"], latency_ms=result["latency_ms"],
                cost=actual_cost, savings=savings, content=result["content"])

st.divider()

# ── Dashboard metrics ─────────────────────────────────────────────────────────
stats = get_stats(conn)
st.subheader("Session metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Requests this session", stats["total"])
c2.metric("Total tokens", f"{stats['tokens']:,}")
c3.metric("Total cost", f"${stats['cost']:.4f}")
c4.metric("Savings vs GPT-4", f"${stats['savings']:.4f}")

if stats["total"] > 0:
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Model distribution")
        fig = px.pie(names=list(stats["models"].keys()),
                     values=list(stats["models"].values()),
                     color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
        fig.update_layout(height=260, margin=dict(t=0,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Complexity tier breakdown")
        fig2 = px.bar(
            x=list(stats["tiers"].keys()), y=list(stats["tiers"].values()),
            color=list(stats["tiers"].keys()),
            color_discrete_map={"simple":"#a6e3a1","medium":"#f9e2af","complex":"#f38ba8"},
            text=list(stats["tiers"].values()),
        )
        fig2.update_layout(height=260, margin=dict(t=0,b=0), showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    history = get_history(conn)
    if history:
        st.subheader("Request history")
        df = pd.DataFrame(history, columns=["Time","Tier","Model","Tokens","Latency(ms)","Savings($)"])
        df["Latency(ms)"] = df["Latency(ms)"].round(1)
        df["Savings($)"] = df["Savings($)"].round(5)
        st.dataframe(df, use_container_width=True, height=200)
else:
    st.info("Send a prompt to see live metrics appear here!")

st.divider()
st.caption("Built with Python · FastAPI · Streamlit · scikit-learn · Groq API (demo) / Ollama (production) · SQLite")
