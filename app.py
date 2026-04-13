"""
Streamlit frontend for the Groupon Ops Intelligence Command Center.
Ties together: data cleaning, AI agent pipeline, backlog tracking, and Slack.

The agent (pipeline.py) uses the Anthropic SDK with tool use to autonomously
analyze ticket data and produce the weekly brief. When no API key is available,
falls back to direct tool calls so the demo still works.
"""

import streamlit as st
import pandas as pd
import os
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

from agent.tools import (
    check_data_quality, flag_anomalies, get_weekly_trends,
    analyze_customer_messages, size_opportunity, generate_brief,
    OPPORTUNITY_META, OPPORTUNITY_ID_MAP, invalidate_cache,
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
        st.info("Using last uploaded dataset -- upload new CSV to refresh")

    st.divider()

    # Agent mode toggle
    has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    use_agent = st.toggle(
        "Use AI Agent (Claude)",
        value=has_api_key,
        disabled=not has_api_key,
        help="When enabled, Claude autonomously decides which tools to call and in what order. "
             "Requires ANTHROPIC_API_KEY in .env. When disabled, runs tools in a fixed sequence."
    )
    if not has_api_key:
        st.caption("Set ANTHROPIC_API_KEY in .env to enable agent mode")

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
        "Agent analysis" if use_agent else "Quality check",
    ]
    if not use_agent:
        steps += ["Trend analysis", "NLP theme extraction", "Anomaly detection",
                   "Opportunity sizing", "Brief generation"]
    steps += ["Backlog updated", "Slack sent"]

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
    invalidate_cache()  # Clear cached data so tools load fresh cleaned data
    mark_step("Data ingested", "done")
    pipeline_log.append({"type": "system", "detail": "Data cleaned and loaded"})

    # Load clean data for backlog
    df = pd.read_csv(ROOT / "data" / "tickets_clean.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])

    if use_agent:
        # -- AGENT MODE: Claude decides what to investigate --
        mark_step("Agent analysis", "running")
        from agent.pipeline import run_agent, get_pipeline_log
        agent_result = run_agent()
        agent_log = get_pipeline_log()
        pipeline_log.extend(agent_log)
        mark_step("Agent analysis", "done")

        # Extract results from agent run for downstream use
        trends = get_weekly_trends()
        nlp = analyze_customer_messages()
        anomalies = flag_anomalies()
        opps = [size_opportunity(name) for name in
                ["chatbot_deflection", "agent_copilot", "urgent_routing", "phone_deflection", "bpo_vendor_b"]]
    else:
        # -- FALLBACK MODE: Fixed sequence without API key --
        mark_step("Quality check", "running")
        dq = check_data_quality()
        mark_step("Quality check", "done")
        pipeline_log.append({"type": "tool_call", "detail": f"check_data_quality -> {dq['row_count']} rows, {len(dq['anomalies_found'])} anomalies"})

        mark_step("Trend analysis", "running")
        trends = get_weekly_trends()
        mark_step("Trend analysis", "done")
        pipeline_log.append({"type": "tool_call", "detail": f"get_weekly_trends -> W{trends['prior_week_num']} vs W{trends['current_week_num']}, {len(trends['flagged_metrics'])} flagged"})

        mark_step("NLP theme extraction", "running")
        nlp = analyze_customer_messages()
        mark_step("NLP theme extraction", "done")
        pipeline_log.append({"type": "tool_call", "detail": f"analyze_customer_messages -> {len(nlp['top_themes'])} themes, {len(nlp['emerging_patterns'])} emerging"})

        mark_step("Anomaly detection", "running")
        anomalies = flag_anomalies()
        mark_step("Anomaly detection", "done")
        pipeline_log.append({"type": "tool_call", "detail": f"flag_anomalies -> {anomalies['count']} anomalies ({sum(1 for a in anomalies['anomalies'] if a['severity']=='critical')} critical)"})

        mark_step("Opportunity sizing", "running")
        opps = []
        for name in ["chatbot_deflection", "agent_copilot", "urgent_routing", "phone_deflection", "bpo_vendor_b"]:
            opp = size_opportunity(name)
            opps.append(opp)
            pipeline_log.append({"type": "tool_call", "detail": f"size_opportunity({name}) -> ${opp['annual_savings']:,}/yr"})
        mark_step("Opportunity sizing", "done")

        mark_step("Brief generation", "running")
        brief_result = generate_brief()
        mark_step("Brief generation", "done")
        pipeline_log.append({"type": "tool_call", "detail": "generate_brief -> saved to output/weekly_brief.md"})

    total_opp = sum(o["annual_savings"] for o in opps)

    # Backlog update
    mark_step("Backlog updated", "running")
    state = update_backlog(df)
    state = fill_open_slots(df, state)
    mark_step("Backlog updated", "done")
    pipeline_log.append({"type": "system", "detail": "Backlog updated with signal values"})

    # Slack
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
    st.session_state["used_agent"] = use_agent

    mode_label = "Agent mode (Claude)" if use_agent else "Fallback mode (no API key)"
    st.success(f"Pipeline complete ({mode_label}) -- total opportunity: ${total_opp:,}/yr")


# ─────────────────────────────────────────────────────────────────
# Main area -- 4 tabs
# ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Weekly Brief",
    "Master Backlog",
    "This Week vs Last Week",
    "Pipeline Log",
])

# -- Tab 1: Weekly Brief --
with tab1:
    brief_path = ROOT / "output" / "weekly_brief.md"
    if brief_path.exists():
        brief_text = brief_path.read_text()

        # Parse the brief into sections for structured rendering
        import re as _re
        _sections = {}
        _current_section = None
        _section_lines = []
        for _line in brief_text.split("\n"):
            if _line.startswith("## "):
                if _current_section is not None:
                    _sections[_current_section] = "\n".join(_section_lines)
                _current_section = _line[3:].strip()
                _section_lines = []
            elif _line.startswith("# "):
                # Title line — render directly
                st.title(_line[2:].strip())
            else:
                _section_lines.append(_line)
        if _current_section:
            _sections[_current_section] = "\n".join(_section_lines)

        # --- Subtitle (Week N | Generated ...) ---
        for _sec_name in list(_sections.keys()):
            if _sec_name.startswith("Week"):
                st.subheader(_sec_name)
                del _sections[_sec_name]
                break

        # --- Executive Summary with KPI callouts ---
        if "Executive Summary" in _sections:
            st.subheader("Executive Summary")
            _summary_text = _sections["Executive Summary"].strip()
            st.markdown(_summary_text)

            # Extract KPI bullets from the summary + other sections
            _kpi_cols = st.columns(3)
            # KPI 1: Total opportunity value
            _total_match = _re.search(r'\$(\d[\d,]+)/yr', _summary_text)
            with _kpi_cols[0]:
                if _total_match:
                    st.metric("Total Opportunity", f"${_total_match.group(1)}/yr")
                else:
                    st.metric("Total Opportunity", "—")
            # KPI 2: Biggest WoW change from the comparison section
            _wow_section = _sections.get("Week-over-Week Comparison", "")
            # Parse WoW table rows: "| Metric | W9 | W10 | Delta | Trend |"
            _wow_rows = [line.strip() for line in _wow_section.split("\n") if line.strip().startswith("|")]
            _wow_deltas = []
            for _wr in _wow_rows:
                _cells = [c.strip() for c in _wr.split("|") if c.strip()]
                if len(_cells) >= 4 and _cells[0] not in ("Metric", "---"):
                    try:
                        _wow_deltas.append((_cells[0], _cells[3]))
                    except (IndexError, ValueError):
                        pass
            if _wow_deltas:
                _biggest = max(_wow_deltas, key=lambda x: abs(float(x[1].rstrip('%'))))
                with _kpi_cols[1]:
                    st.metric("Biggest WoW Shift", _biggest[0], _biggest[1])
            # KPI 3: Most urgent action
            _actions_section = _sections.get("Recommended Actions This Week", "")
            _immediate_match = _re.search(r'IMMEDIATE.*?:\s*(.+?)(?:\.|—)', _actions_section)
            with _kpi_cols[2]:
                if _immediate_match:
                    _urgent_action = _immediate_match.group(1).strip()[:50]
                    st.metric("Most Urgent Action", _urgent_action + "..." if len(_immediate_match.group(1).strip()) > 50 else _urgent_action)
                else:
                    st.metric("Most Urgent Action", "None flagged")
            st.divider()

        # --- Top 5 Issues This Week (structured cards) ---
        if "Top 5 Issues This Week" in _sections:
            st.subheader("Top 5 Issues This Week")
            _issues_text = _sections["Top 5 Issues This Week"].strip()
            # Parse numbered issues into individual blocks
            _issue_blocks = _re.split(r"\n(?=\d+\.\s)", _issues_text)
            for _block in _issue_blocks:
                _block = _block.strip()
                if not _block:
                    continue
                # Skip the "Ranked by..." subtitle
                if _block.startswith("*Ranked"):
                    st.caption(_block.strip("*"))
                    continue
                # Parse issue: "1. **Name** — $X/yr (range: ...)\n   Root cause: ...\n   ..."
                _header_match = _re.match(r"(\d+)\.\s+\*\*(.+?)\*\*\s*—\s*(.+?)(?:\n|$)", _block)
                if _header_match:
                    _num = _header_match.group(1)
                    _name = _header_match.group(2)
                    _value = _header_match.group(3).split("\n")[0].strip()
                    # Extract detail lines
                    _detail_lines = _block[_header_match.end():].strip().split("\n")
                    _root_cause = ""
                    _action = ""
                    _owner_timeline = ""
                    _kpi = ""
                    _confidence = ""
                    for _dl in _detail_lines:
                        _dl = _dl.strip()
                        if _dl.startswith("Root cause:"):
                            _root_cause = _dl[len("Root cause:"):].strip()
                        elif _dl.startswith("Action:"):
                            _action = _dl[len("Action:"):].strip()
                        elif _dl.startswith("Owner:"):
                            _owner_timeline = _dl
                        elif _dl.startswith("KPI target:"):
                            _kpi = _dl[len("KPI target:"):].strip()
                        elif _dl.startswith("Confidence:"):
                            _confidence = _dl[len("Confidence:"):].strip()

                    _safe_value = _value.replace("$", "\\$")
                    with st.expander(f"**{_num}. {_name}** — {_safe_value}", expanded=True):
                        if _root_cause:
                            st.markdown(f"**Root cause:** {_root_cause}")
                        if _action:
                            st.markdown(f"**Action:** {_action}")
                        if _owner_timeline:
                            st.markdown(f"**{_owner_timeline}**")
                        if _kpi:
                            st.markdown(f"**KPI target:** {_kpi}")
                        if _confidence:
                            st.caption(f"Confidence: {_confidence}")
                else:
                    st.markdown(_block)
            st.divider()

        # --- Week-over-Week Comparison (already a markdown table — render as-is) ---
        if "Week-over-Week Comparison" in _sections:
            st.subheader("Week-over-Week Comparison")
            st.markdown(_sections["Week-over-Week Comparison"].strip())
            st.divider()

        # --- Recommended Actions (table) ---
        if "Recommended Actions This Week" in _sections:
            st.subheader("Recommended Actions This Week")
            _actions_text = _sections["Recommended Actions This Week"].strip()
            # Parse: "- **TIMELINE**: Action text — Owner: Name"
            _action_items = _re.findall(
                r'-\s+\*\*(.+?)\*\*:\s*(.+?)\s*(?:—|--|--)\s*Owner:\s*(.+?)\s*$',
                _actions_text,
                _re.MULTILINE,
            )
            if _action_items:
                for _timeline, _action_desc, _owner in _action_items:
                    _priority_icon = "🔴" if "IMMEDIATE" in _timeline else "🟠" if "SHORT" in _timeline else "🔵"
                    st.markdown(
                        f"{_priority_icon} **{_timeline}**  \n"
                        f"{_action_desc.strip()}  \n"
                        f"*Owner: {_owner.strip()}*"
                    )
                    st.caption("")  # spacing between actions
            else:
                st.markdown(_actions_text)
            st.divider()

        # --- Watch List — Emerging Patterns (table) ---
        _watch_key = [k for k in _sections if "Watch List" in k]
        if _watch_key:
            st.subheader("Watch List — Emerging Patterns")
            _watch_text = _sections[_watch_key[0]].strip()
            # Parse into rows: "- **theme**: grew X% WoW (...)\n  Why it matters: ..."
            _watch_items = _re.split(r"\n(?=-\s\*\*)", _watch_text)
            _watch_rows = []
            for _item in _watch_items:
                _item = _item.strip()
                if not _item:
                    continue
                _m = _re.match(
                    r"-\s+\*\*(.+?)\*\*:\s+(?:grew|moved)\s+([\d.+%-]+)%?\s+(?:WoW\s+)?\((.+?)\)",
                    _item,
                )
                _why_match = _re.search(r"Why it matters:\s*(.+)", _item)
                if _m:
                    _watch_rows.append({
                        "Pattern": _m.group(1),
                        "Change": _m.group(2) if "%" in _m.group(2) else f"+{_m.group(2)}%",
                        "Detail": _m.group(3),
                        "Why It Matters": _why_match.group(1).strip() if _why_match else "",
                    })
            if _watch_rows:
                st.dataframe(
                    pd.DataFrame(_watch_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.markdown(_watch_text)
            st.divider()

        # --- Data Quality Notes (table) ---
        if "Data Quality Notes" in _sections:
            st.subheader("Data Quality Notes")
            _dq_text = _sections["Data Quality Notes"].strip()
            _dq_lines = [l.strip().lstrip("- ") for l in _dq_text.split("\n") if l.strip().startswith("-")]
            if _dq_lines:
                _dq_rows = []
                for _dl in _dq_lines:
                    # Try to split on ":" for label/value
                    if ": " in _dl:
                        _parts = _dl.split(": ", 1)
                        _severity = "🔴 Critical" if "CRITICAL" in _parts[0] else \
                                    "🟠 Warning" if "WARNING" in _parts[0] else "ℹ️ Info"
                        _label = _parts[0].replace("CRITICAL", "").replace("WARNING", "").strip()
                        if not _label:
                            _dq_rows.append({
                                "Severity": _severity,
                                "Note": _parts[1],
                                "Detail": "",
                            })
                        else:
                            _dq_rows.append({
                                "Severity": _severity,
                                "Note": _label,
                                "Detail": _parts[1],
                            })
                    else:
                        _dq_rows.append({"Severity": "ℹ️ Info", "Note": _dl, "Detail": ""})
                st.dataframe(
                    pd.DataFrame(_dq_rows),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.markdown(_dq_text)

        mod_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(brief_path.stat().st_mtime))
        st.caption(f"Last generated: {mod_time}")

        # Download button for the brief
        st.download_button(
            label="📥 Download Brief (Markdown)",
            data=brief_text,
            file_name=f"ops_intelligence_brief_{mod_time.replace(' ', '_').replace(':', '')}.md",
            mime="text/markdown",
        )
    else:
        st.info("No brief generated yet. Click **Run Analysis** in the sidebar.")

# -- Tab 2: Master Backlog --
with tab2:
    st.caption(
        "The **Master Backlog** tracks the top 5 operational improvement opportunities ranked by "
        "estimated annual savings. Each opportunity has a **Signal** (the current metric value measured "
        "this week) and a **Target** (the goal to hit). **Trend** shows whether the signal is moving "
        "toward or away from the target compared to last week. An opportunity is auto-resolved once "
        "the signal stays at target for 2 consecutive weeks."
    )
    backlog = _load_state()
    if backlog and backlog.get("opportunities"):
        opps_list = sorted(backlog["opportunities"], key=lambda x: x["annual_value"], reverse=True)

        # Human-readable metric labels for the Signal column
        signal_labels = {
            "chatbot_deflection": ("Chatbot %", lambda v: f"{v:.1%}"),
            "agent_copilot": ("Contacts/ticket", lambda v: f"{v:.1f}"),
            "urgent_routing": ("Urgent→chatbot", lambda v: f"{int(v)}"),
            "phone_deflection": ("Phone %", lambda v: f"{v:.1%}"),
            "bpo_vendor_b": ("Vendor B CSAT", lambda v: f"{v:.2f}"),
        }

        rows = []
        for i, opp in enumerate(opps_list, 1):
            sig_def = OPPORTUNITY_SIGNALS.get(opp["id"])
            raw_current = opp["signal_history"][-1]["value"] if opp.get("signal_history") else None
            weeks_tracked = len(opp.get("signal_history", []))

            # Format signal and target with readable labels
            label_info = signal_labels.get(opp["id"])
            if label_info and raw_current is not None:
                metric_name, fmt = label_info
                signal_str = f"{fmt(raw_current)}"
            elif raw_current is not None:
                signal_str = f"{raw_current:.2f}"
            else:
                signal_str = "—"

            if sig_def and label_info:
                _, fmt = label_info
                direction_word = "≤" if sig_def["target_direction"] == "below" else "≥"
                target_str = f"{direction_word} {fmt(sig_def['target'])}"
            elif sig_def:
                target_str = f"{sig_def['target']}"
            else:
                target_str = "—"

            # Trend arrow
            if len(opp.get("signal_history", [])) >= 2:
                prev = opp["signal_history"][-2]["value"]
                curr = opp["signal_history"][-1]["value"]
                if sig_def and sig_def["target_direction"] == "below":
                    trend = "improving" if curr < prev else ("worsening" if curr > prev else "flat")
                else:
                    trend = "improving" if curr > prev else ("worsening" if curr < prev else "flat")
            else:
                trend = "new"

            trend_labels = {
                "improving": "📈 Improving",
                "worsening": "📉 Worsening",
                "flat": "➡️ Flat",
                "new": "🆕 New",
            }

            rows.append({
                "Rank": i,
                "Opportunity": opp["title"],
                "Status": opp["status"].replace("_", " ").title(),
                "Signal (now)": signal_str,
                "Target": target_str,
                "Trend": trend_labels.get(trend, "—"),
                "Owner": opp["owner"],
                "Annual Value": f"${opp['annual_value']:,}",
                "Weeks Tracked": weeks_tracked,
            })

        df_backlog = pd.DataFrame(rows)

        # Color rows by status
        def color_status(row):
            colors = {
                "Resolved": "background-color: #d4edda",
                "In Progress": "background-color: #fff3cd",
                "Stalled": "background-color: #f8d7da",
            }
            return [colors.get(row["Status"], "")] * len(row)

        st.dataframe(
            df_backlog.style.apply(color_status, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Signal History Trend Charts
        st.subheader("Signal History")
        st.caption("Tracking how each opportunity's signal moves toward its target over time. "
                    "The dashed line shows the target value.")
        # Layout: row of 3, then row of 2 (or fewer)
        _opps_with_history = [o for o in opps_list if o.get("signal_history")]
        for _row_start in range(0, len(_opps_with_history), 3):
            _row_opps = _opps_with_history[_row_start:_row_start + 3]
            _chart_cols = st.columns(3)
            for _ci, opp in enumerate(_row_opps):
                _history = opp.get("signal_history", [])
                sig_def = OPPORTUNITY_SIGNALS.get(opp["id"])
                _hist_values = [h["value"] for h in _history]
                # Use week labels from history if available
                _hist_weeks = [str(h.get("week", f"W{i+1}")) for i, h in enumerate(_history)]

                with _chart_cols[_ci]:
                    _short_title = opp["title"].replace("Fix ", "").replace("Expand ", "")
                    label_info = signal_labels.get(opp["id"])
                    st.caption(f"**{_short_title}**")
                    if len(_hist_values) >= 2:
                        _line_data = pd.DataFrame(
                            {"Signal": _hist_values},
                            index=_hist_weeks,
                        )
                        if sig_def:
                            _line_data["Target"] = sig_def["target"]
                        st.line_chart(_line_data, height=180)
                    else:
                        if label_info:
                            _, fmt = label_info
                            st.metric("Current", fmt(_hist_values[0]))
                        else:
                            st.metric("Current", f"{_hist_values[0]:.2f}")
                        if sig_def:
                            st.caption(f"Target: {sig_def['target']}")

        st.divider()

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
                st.write(f"- **{c['title']}** -- est. ${c['estimated_annual_value']:,}/yr [{c['source']}]")
    else:
        st.info("No backlog data yet. Click **Run Analysis** in the sidebar.")

# -- Tab 3: This Week vs Last Week --
with tab3:
    trends = st.session_state.get("trends")
    nlp = st.session_state.get("nlp")
    opps = st.session_state.get("opps")

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
                    delta_color="inverse" if key in ["avg_cost_per_ticket", "escalation_rate", "abandonment_rate", "phone_ticket_pct", "avg_contacts"] else "normal",
                )

        st.divider()

        # -- Charts --
        if opps:
            st.subheader("Opportunity Sizing -- Annual Savings")
            # Use short labels so the y-axis doesn't truncate names
            short_labels = {
                "Expand chatbot deflection": "Chatbot deflection",
                "AI agent co-pilot": "Agent co-pilot",
                "Fix urgent ticket routing": "Urgent routing",
                "Phone to chat deflection": "Phone deflection",
                "BPO Vendor B quality intervention": "Vendor B quality",
            }
            opp_chart_data = pd.DataFrame([
                {
                    "Opportunity": short_labels.get(o["opportunity"], o["opportunity"]),
                    "Annual Savings ($)": o["annual_savings"],
                }
                for o in sorted(opps, key=lambda x: x["annual_savings"], reverse=True)
            ])
            st.bar_chart(opp_chart_data, x="Opportunity", y="Annual Savings ($)", horizontal=True)

        # WoW trend comparison table
        st.subheader("Key Metrics -- Week-over-Week")
        wow_rows = []
        wow_metrics = ["avg_cost_per_ticket", "avg_csat", "resolution_rate", "escalation_rate"]
        wow_labels = {
            "avg_cost_per_ticket": "Avg Cost ($)",
            "avg_csat": "CSAT",
            "resolution_rate": "Resolution %",
            "escalation_rate": "Escalation %",
        }
        for key in wow_metrics:
            data = comparison.get(key, {})
            wow_rows.append({
                "Metric": wow_labels.get(key, key),
                f"W{prior_w}": data.get("prior_week", 0),
                f"W{curr_w}": data.get("current_week", 0),
                "Delta %": f"{data.get('pct_change', 0):+.1f}%",
                "Direction": data.get("direction", "flat").capitalize(),
            })
        wow_df = pd.DataFrame(wow_rows)
        st.dataframe(wow_df, use_container_width=True, hide_index=True)

        # Team performance heatmap
        clean_path = ROOT / "data" / "tickets_clean.csv"
        if clean_path.exists():
            st.subheader("Team Performance Heatmap")
            df_clean = pd.read_csv(clean_path)
            team_perf = df_clean.groupby("assigned_team").agg(
                avg_csat=("csat_score", "mean"),
                resolution_rate=("is_resolved", lambda x: round(x.mean() * 100, 1)),
                avg_contacts=("contacts_per_ticket", "mean"),
                avg_cost=("cost_usd", "mean"),
            ).round(2)

            def highlight_csat(val):
                if val < 3.2:
                    return "background-color: #f8d7da"
                elif val > 3.8:
                    return "background-color: #d4edda"
                return ""

            def highlight_cost(val):
                if val > 4.0:
                    return "background-color: #f8d7da"
                elif val < 2.0:
                    return "background-color: #d4edda"
                return ""

            styled = team_perf.style.map(
                highlight_csat, subset=["avg_csat"]
            ).map(
                highlight_cost, subset=["avg_cost"]
            )
            st.dataframe(styled, use_container_width=True)

        st.divider()

        # NLP themes
        if nlp:
            col_a, col_b = st.columns(2)

            with col_a:
                st.subheader(f"Top Customer Message Themes (W{nlp.get('latest_week', '?')})")
                latest_themes = nlp.get("latest_week_themes", nlp.get("top_themes", []))
                for theme in latest_themes[:5]:
                    st.write(f"- **{theme['theme']}** — {theme['tickets']} tickets ({theme['pct_of_total']}%)")

            with col_b:
                st.subheader("Emerging Patterns (Week-over-Week)")
                for pattern in nlp.get("emerging_patterns", [])[:5]:
                    growth = pattern["growth_pct"]
                    icon = "🔴" if growth > 40 else "🟠" if growth > 25 else "🔵"
                    st.write(
                        f"- **{pattern['theme']}** — "
                        f"{icon} +{growth:.0f}% "
                        f"(W{nlp['prior_week']}: {pattern['w_prior_count']} → "
                        f"W{nlp['latest_week']}: {pattern['w_latest_count']})"
                    )
    else:
        st.info("No trend data yet. Click **Run Analysis** in the sidebar.")

# -- Tab 4: Pipeline Log --
with tab4:
    pipeline_log = st.session_state.get("pipeline_log")
    used_agent = st.session_state.get("used_agent", False)

    if pipeline_log:
        if used_agent:
            st.subheader("Agent Pipeline Trace (Claude)")
            st.caption("The agent autonomously decided which tools to call and in what order. "
                       "Reasoning steps show the agent's thought process between tool calls.")
        else:
            st.subheader("Pipeline Trace (Fallback Mode)")
            st.caption("Ran without API key -- tools called in fixed sequence. "
                       "Enable agent mode with ANTHROPIC_API_KEY to see autonomous reasoning.")

        for i, entry in enumerate(pipeline_log):
            entry_type = entry["type"]
            detail = entry["detail"]
            # Escape $ signs so Streamlit doesn't interpret them as LaTeX
            safe_detail = detail.replace("$", "\\$") if isinstance(detail, str) else detail

            if entry_type == "tool_call":
                st.code(f"[TOOL] {detail}", language=None)
            elif entry_type == "tool_result":
                st.code(f"[RESULT] {detail}", language=None)
            elif entry_type == "system":
                st.info(f"{safe_detail}")
            elif entry_type == "thinking":
                st.write(f"**Agent reasoning:** {safe_detail}")
            elif entry_type == "final":
                st.success(f"**Final:** {safe_detail}")
            elif entry_type == "guardrail":
                st.info(f"**Guardrail:** {safe_detail}")
            else:
                st.write(safe_detail)
    else:
        # Try to load from agent's pipeline log
        try:
            from agent.pipeline import get_pipeline_log
            log = get_pipeline_log()
            if log:
                st.subheader("Agent Pipeline Trace (from last agent run)")
                for entry in log:
                    entry_type = entry["type"]
                    detail = entry["detail"]
                    safe_detail = detail.replace("$", "\\$") if isinstance(detail, str) else detail
                    if entry_type == "thinking":
                        st.write(f"**Agent reasoning:** {safe_detail}")
                    elif entry_type in ("tool_call", "tool_result"):
                        st.code(f"[{entry_type.upper()}] {detail}", language=None)
                    elif entry_type == "guardrail":
                        st.info(f"**Guardrail:** {safe_detail}")
                    else:
                        st.write(safe_detail)
            else:
                st.info("No pipeline log yet. Click **Run Analysis** in the sidebar.")
        except Exception:
            st.info("No pipeline log yet. Click **Run Analysis** in the sidebar.")
