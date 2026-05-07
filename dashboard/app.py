"""
dashboard/app.py
Streamlit dashboard showing cost savings, model distribution, and latency trends.
Run: streamlit run dashboard/app.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.db.database import Database
from app.utils.config import get_setting

st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

REFRESH = int(get_setting("dashboard.refresh_interval", 30))

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card { background: #1e1e2e; border-radius: 12px; padding: 1rem; }
    .big-number  { font-size: 2.5rem; font-weight: 700; color: #cba6f7; }
    .stMetric > div { background: #1e1e2e; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    return Database()


def load_data(db: Database):
    stats = db.get_stats()
    recent = db.get_recent_requests(200)
    latency = db.get_latency_trend(200)
    return stats, recent, latency


def main():
    st.title("🚀 LLM Cost Autopilot — Dashboard")
    st.caption("Zero-cost local LLM routing with intelligent complexity classification")

    db = get_db()
    stats, recent, latency_data = load_data(db)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Requests",   f"{stats['total_requests']:,}")
    c2.metric("Total Tokens",     f"{stats['total_tokens']:,}")
    c3.metric("Total Cost",       f"${stats['total_cost']:.4f}")
    c4.metric("Est. Savings vs GPT-4", f"${stats['total_savings']:.2f}")
    c5.metric("Escalation Rate",  f"{stats['escalation_rate']*100:.1f}%")

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Model Distribution ────────────────────────────────────────────────────
    with col_left:
        st.subheader("🤖 Model Distribution")
        dist = stats.get("model_distribution", {})
        if dist:
            fig = px.pie(
                names=list(dist.keys()),
                values=list(dist.values()),
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data yet — send some requests!")

    # ── Latency Trend ─────────────────────────────────────────────────────────
    with col_right:
        st.subheader("⚡ Latency Trend")
        if latency_data:
            df_lat = pd.DataFrame(latency_data)
            df_lat["timestamp"] = pd.to_datetime(df_lat["timestamp"])
            df_lat = df_lat.sort_values("timestamp")
            fig2 = px.line(
                df_lat,
                x="timestamp",
                y="latency_ms",
                color="model_used",
                labels={"latency_ms": "Latency (ms)", "timestamp": "Time"},
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig2.update_layout(margin=dict(t=0, b=0), height=300)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No latency data yet.")

    st.divider()

    # ── Savings Projection ────────────────────────────────────────────────────
    st.subheader("💰 Cost Savings Analysis")
    if recent:
        df = pd.DataFrame(recent)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        df["cumulative_savings"] = df["estimated_savings"].cumsum()
        df["cumulative_cost"] = df["cost"].cumsum()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["timestamp"], y=df["cumulative_savings"],
            name="Cumulative Savings", fill="tozeroy",
            line=dict(color="#a6e3a1"), fillcolor="rgba(166,227,161,0.2)",
        ))
        fig3.add_trace(go.Scatter(
            x=df["timestamp"], y=df["cumulative_cost"],
            name="Actual Cost (local)", line=dict(color="#cba6f7"),
        ))
        fig3.update_layout(height=280, margin=dict(t=0, b=0),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Complexity Breakdown ──────────────────────────────────────────────────
    st.divider()
    st.subheader("🧠 Complexity Tier Breakdown")
    if recent:
        df = pd.DataFrame(recent)
        tier_counts = df["complexity_tier"].value_counts().reset_index()
        tier_counts.columns = ["tier", "count"]
        fig4 = px.bar(
            tier_counts, x="tier", y="count",
            color="tier",
            color_discrete_map={"simple": "#a6e3a1", "medium": "#f9e2af", "complex": "#f38ba8"},
            text="count",
        )
        fig4.update_layout(height=250, margin=dict(t=0, b=0), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Recent Requests Table ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Recent Requests")
    if recent:
        df = pd.DataFrame(recent)
        display_cols = ["timestamp", "complexity_tier", "model_used",
                        "total_tokens", "latency_ms", "estimated_savings", "quality_score", "escalated"]
        df_show = df[display_cols].head(20).copy()
        df_show["latency_ms"] = df_show["latency_ms"].round(1)
        df_show["estimated_savings"] = df_show["estimated_savings"].round(4)
        st.dataframe(df_show, use_container_width=True, height=300)
    else:
        st.info("No requests logged yet.")

    # Auto-refresh
    st.sidebar.markdown(f"🔄 Auto-refreshes every **{REFRESH}s**")
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_resource.clear()
        st.rerun()

    time.sleep(REFRESH)
    st.rerun()


if __name__ == "__main__":
    main()
