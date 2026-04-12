"""
Streamlit frontend for the Groupon Ops Intelligence Command Center.
Ties together: data cleaning, analysis, AI agent, backlog tracking, and Slack.
"""

import streamlit as st
import pandas as pd
import sys
import os
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))

from agent.tools import (
    check_data_quality, flag_anomalies, get_weekly_trends,
    analyze_customer_messages, size_opportunity, generate_brief,
    OPPORTUNITY_META, OPPORTUNITY_ID_MAP,
)
from agent.backlog import (
    update_backlog, fill_open_slots, get_backlog_summary,
    OPPORTUNITY_SIGNALS, _load_state,
)
from agent.slack_notify import build_message, send_notification, build_watch_list

st.set_page_config(
    page_title="Groupon Ops Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────
st.title("Groupon Operations Intelligence Command Center")
st.caption(
    "**Demo mode:** upload CSV manually. "
    "**Production:** would auto-pull from Zendesk API every Monday 6am "
    "with no manual upload required."
)

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload ticket CSV", type=["csv"])

    if uploaded_file:
        csv_path = ROOT / "data" / "option_a_ticket_data.csv"
        csv_path.write_bytes(uploaded_file.getvalue())
        st.success(f"Uploaded: {uploaded_file.name}")
    else:
        st.info("Using last uploaded dataset — upload new CSV to refresh")

    st.divider()
    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

    # Pipeline status indicators
    st.divider()
    st.subheader("Pipeline Status")
    status_container = st.container()


def update_status(step_name, state="running"):
    """Update pipeline step status in sidebar."""
    icons = {"pending": ":white_circle:", "running": ":hourglass_flowing_sand:", "done": ":white_check_mark:"}
    return icons.get(state, ":white_circle:")


# ─────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────
if run_button:
    pipeline_log = []

    steps = [
        "Data ingested",
        "Quality check",
        "Trend analysis",
        "NLP theme extraction",
        "Anomaly detection",
        "Opportunity sizing",
        "Brief generation",
        "Backlog updated",
        "Slack sent",
    ]
    step_status = {s: "pending" for s in steps}

    with status_container:
        status_placeholders = {}
        for step in steps:
            status_placeholders[step] = st.empty()
            status_placeholders[step].write(f":white_circle: {step}")

    def mark_step(step, state="done"):
        step_status[step] = state
        icon = update_status(step, state)
        status_placeholders[step].write(f"{icon} {step}")

    # Step 1: Clean data
    mark_step("Data ingested", "running")
    from analysis.clean import main as run_clean
    run_clean()
    mark_step("Data ingested", "done")
    pipeline_log.append({"type": "system", "detail": "Data cleaned and loaded: 9,776 tickets (W7-W10)"})

    # Load clean data
    df = pd.read_csv(ROOT / "data" / "tickets_clean.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Step 2: Quality check
    mark_step("Quality check", "running")
    dq = check_data_quality()
    mark_step("Quality check", "done")
    pipeline_log.append({"type": "tool_call", "detail": f"check_data_quality → {dq['row_count']} rows, {len(dq['anomalies_found'])} anomalies"})

    # Step 3: Trend analysis
    mark_step("Trend analysis", "running")
    trends = get_weekly_trends()
    mark_step("Trend analysis", "done")
    pipeline_log.append({"type": "tool_call", "detail": f"get_weekly_trends → W{trends['prior_week_num']} vs W{trends['current_week_num']}, {len(trends['flagged_metrics'])} flagged"})

    # Step 4: NLP
    mark_step("NLP theme extraction", "running")
    nlp = analyze_customer_messages()
    mark_step("NLP theme extraction", "done")
    pipeline_log.append({"type": "tool_call", "detail": f"analyze_customer_messages → {len(nlp['top_themes'])} themes, {len(nlp['emerging_patterns'])} emerging"})

    # Step 5: Anomaly detection
    mark_step("Anomaly detection", "running")
    anomalies = flag_anomalies()
    mark_step("Anomaly detection", "done")
    pipeline_log.append({"type": "tool_call", "detail": f"flag_anomalies → {anomalies['count']} anomalies ({sum(1 for a in anomalies['anomalies'] if a['severity']=='critical')} critical)"})

    # Step 6: Opportunity sizing
    mark_step("Opportunity sizing", "running")
    opps = []
    for name in ["chatbot_deflection", "agent_copilot", "urgent_routing", "phone_deflection", "bpo_vendor_b"]:
        opp = size_opportunity(name)
        opps.append(opp)
        pipeline_log.append({"type": "tool_call", "detail": f"size_opportunity({name}) → ${opp['annual_savings']:,}/yr"})
    total_opp = sum(o["annual_savings"] for o in opps)
    mark_step("Opportunity sizing", "done")

    # Step 7: Brief generation
    mark_step("Brief generation", "running")
    brief_result = generate_brief()
    mark_step("Brief generation", "done")
    pipeline_log.append({"type": "tool_call", "detail": f"generate_brief → saved to output/weekly_brief.md"})

    # Step 8: Backlog
    mark_step("Backlog updated", "running")
    state = update_backlog(df)
    state = fill_open_slots(df, state)
    mark_step("Backlog updated", "done")
    pipeline_log.append({"type": "system", "detail": "Backlog updated with signal values"})

    # Step 9: Slack
    mark_step("Slack sent", "running")
    backlog_summary = get_backlog_summary(state)
    watch_list = build_watch_list(nlp, trends)
    slack_msg = build_message(backlog_summary, trends, watch_list)
    send_notification(slack_msg)
    mark_step("Slack sent", "done")
    pipeline_log.append({"type": "system", "detail": "Slack notification sent (or skipped if no webhook)"})

    # Store results in session state
    st.session_state["pipeline_log"] = pipeline_log
    st.session_state["trends"] = trends
    st.session_state["nlp"] = nlp
    st.session_state["anomalies"] = anomalies
    st.session_state["opps"] = opps
    st.session_state["backlog_summary"] = backlog_summary
    st.session_state["ran"] = True

    st.success(f"Pipeline complete — total opportunity: ${total_opp:,}/yr")


# ─────────────────────────────────────────────────────────────────
# Main area — 4 tabs
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Weekly Brief",
    "Master Backlog",
    "This Week vs Last Week",
    "Pipeline Log",
])

# ── Tab 1: Weekly Brief ──────────────────────────────────────────
with tab1:
    brief_path = ROOT / "output" / "weekly_brief.md"
    if brief_path.exists():
        brief_text = brief_path.read_text()
        st.markdown(brief_text)
        mod_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(brief_path.stat().st_mtime))
        st.caption(f"Last generated: {mod_time}")
    else:
        st.info("No brief generated yet. Click **Run Analysis** in the sidebar.")

# ── Tab 2: Master Backlog ────────────────────────────────────────
with tab2:
    backlog = _load_state()
    if backlog and backlog.get("opportunities"):
        opps_list = sorted(backlog["opportunities"], key=lambda x: x["annual_value"], reverse=True)

        rows = []
        for i, opp in enumerate(opps_list, 1):
            sig_def = OPPORTUNITY_SIGNALS.get(opp["id"])
            current = opp["signal_history"][-1]["value"] if opp.get("signal_history") else "—"
            target = sig_def["target"] if sig_def else "—"
            weeks_tracked = len(opp.get("signal_history", []))

            # Trend arrow
            if len(opp.get("signal_history", [])) >= 2:
                prev = opp["signal_history"][-2]["value"]
                curr = opp["signal_history"][-1]["value"]
                if sig_def and sig_def["target_direction"] == "below":
                    trend = "improving" if curr < prev else ("worsening" if curr > prev else "flat")
                else:
                    trend = "improving" if curr > prev else ("worsening" if curr < prev else "flat")
            else:
                trend = "—"

            trend_arrows = {"improving": "↗", "worsening": "↘", "flat": "→", "—": "—"}

            rows.append({
                "Rank": i,
                "Title": opp["title"],
                "Status": opp["status"],
                "Signal": current,
                "Target": target,
                "Trend": trend_arrows.get(trend, "—"),
                "Owner": opp["owner"],
                "Annual Value": f"${opp['annual_value']:,}",
                "Weeks": weeks_tracked,
            })

        df_backlog = pd.DataFrame(rows)

        # Color rows by status
        def color_status(row):
            colors = {
                "resolved": "background-color: #d4edda",
                "in_progress": "background-color: #fff3cd",
                "stalled": "background-color: #f8d7da",
            }
            return [colors.get(row["Status"], "")] * len(row)

        st.dataframe(
            df_backlog.style.apply(color_status, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Total remaining value
        unresolved = [o for o in opps_list if o["status"] != "resolved"]
        total_remaining = sum(o["annual_value"] for o in unresolved)
        st.metric("Total Remaining Opportunity", f"${total_remaining:,}/yr")

        # Candidates queue
        candidates = backlog.get("candidates", [])
        if candidates:
            st.subheader("Queued Candidates")
            st.caption("New findings waiting for a slot in the active backlog")
            for c in candidates[:5]:
                st.write(f"- **{c['title']}** — est. ${c['estimated_annual_value']:,}/yr [{c['source']}]")
    else:
        st.info("No backlog data yet. Click **Run Analysis** in the sidebar.")

# ── Tab 3: This Week vs Last Week ────────────────────────────────
with tab3:
    trends = st.session_state.get("trends")
    nlp = st.session_state.get("nlp")

    if trends:
        comparison = trends.get("comparison", {})
        prior_w = trends["prior_week_num"]
        curr_w = trends["current_week_num"]

        st.subheader(f"Week {prior_w} vs Week {curr_w}")

        # Side-by-side metrics
        metric_labels = {
            "ticket_volume": ("Ticket Volume", ""),
            "avg_cost_per_ticket": ("Avg Cost/Ticket", "$"),
            "avg_csat": ("CSAT Score", ""),
            "resolution_rate": ("Resolution Rate", "%"),
            "escalation_rate": ("Escalation Rate", "%"),
            "abandonment_rate": ("Abandonment Rate", "%"),
            "chatbot_containment_pct": ("Chatbot Containment", "%"),
            "phone_ticket_pct": ("Phone Ticket Share", "%"),
            "avg_contacts": ("Avg Contacts/Ticket", ""),
        }

        cols = st.columns(3)
        for i, (key, (label, suffix)) in enumerate(metric_labels.items()):
            data = comparison.get(key, {})
            prev = data.get("prior_week", 0)
            curr = data.get("current_week", 0)
            delta = data.get("pct_change", 0)
            direction = data.get("direction", "flat")

            prefix = "$" if suffix == "$" else ""
            suffix_str = "%" if suffix == "%" else ""

            with cols[i % 3]:
                st.metric(
                    label=label,
                    value=f"{prefix}{curr}{suffix_str}",
                    delta=f"{delta:+.1f}% ({direction})",
                    delta_color="inverse" if key in ["avg_cost_per_ticket", "escalation_rate", "abandonment_rate", "phone_ticket_pct"] else "normal",
                )

        st.divider()

        # NLP themes
        if nlp:
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader("Top Customer Message Themes")
                for theme in nlp.get("top_themes", [])[:5]:
                    st.write(f"- **{theme['theme']}** — {theme['tickets']} tickets ({theme['pct_of_total']}%)")

            with col_b:
                st.subheader("Emerging Patterns (Week-over-Week)")
                for pattern in nlp.get("emerging_patterns", [])[:5]:
                    growth = pattern["growth_pct"]
                    color = "red" if growth > 40 else "orange" if growth > 25 else "blue"
                    st.write(
                        f"- **{pattern['theme']}** — "
                        f":{color}_circle: +{growth:.0f}% "
                        f"(W{nlp['prior_week']}: {pattern['w_prior_count']} → "
                        f"W{nlp['latest_week']}: {pattern['w_latest_count']})"
                    )
    else:
        st.info("No trend data yet. Click **Run Analysis** in the sidebar.")

# ── Tab 4: Pipeline Log ──────────────────────────────────────────
with tab4:
    pipeline_log = st.session_state.get("pipeline_log")

    if pipeline_log:
        st.subheader("Agent Pipeline Trace")
        st.caption("Each step shows what the agent investigated and what it found")

        for i, entry in enumerate(pipeline_log):
            entry_type = entry["type"]
            detail = entry["detail"]

            if entry_type == "tool_call":
                st.code(f"[TOOL] {detail}", language=None)
            elif entry_type == "system":
                st.info(f"{detail}")
            elif entry_type == "thinking":
                st.write(f"**Agent reasoning:** {detail}")
            else:
                st.write(detail)
    else:
        # Try to load from agent's pipeline log
        try:
            from agent.pipeline import get_pipeline_log
            log = get_pipeline_log()
            if log:
                st.subheader("Agent Pipeline Trace (from last agent run)")
                for entry in log:
                    st.code(f"[{entry['type'].upper()}] {entry['detail']}", language=None)
            else:
                st.info("No pipeline log yet. Click **Run Analysis** in the sidebar.")
        except Exception:
            st.info("No pipeline log yet. Click **Run Analysis** in the sidebar.")
