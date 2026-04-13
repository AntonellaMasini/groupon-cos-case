"""
Backlog tracking system for Ops Intelligence.
Tracks opportunities via signal metrics, auto-resolves when targets are hit,
detects stalled items, and promotes new priorities from anomaly detection + NLP.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

from agent.tools import flag_anomalies, analyze_customer_messages
STATE_PATH = Path(__file__).resolve().parent / "backlog_state.json"

MAX_PRIORITIES = 5


# ─────────────────────────────────────────────────────────────────
# Opportunity signal definitions
# ─────────────────────────────────────────────────────────────────
OPPORTUNITY_SIGNALS = {
    "urgent_routing": {
        "title": "Fix urgent ticket routing",
        "metric": "urgent_chatbot_count",
        "baseline": 244,
        "target": 10,
        "target_direction": "below",  # metric should go below target
        "annual_value": 35000,
        "owner": "Head of CX Ops",
        "consecutive_weeks_needed": 2,
    },
    "chatbot_deflection": {
        "title": "Expand chatbot deflection",
        "metric": "chatbot_pct",
        "baseline": 0.279,
        "target": 0.43,
        "target_direction": "above",  # metric should go above target
        "annual_value": 68000,
        "owner": "AI Product Lead",
        "consecutive_weeks_needed": 2,
    },
    "phone_deflection": {
        "title": "Phone to chat deflection",
        "metric": "phone_pct",
        "baseline": 0.201,
        "target": 0.161,
        "target_direction": "below",
        "annual_value": 20000,
        "owner": "Channel Strategy Lead",
        "consecutive_weeks_needed": 2,
    },
    "bpo_vendor_b": {
        "title": "BPO Vendor B quality",
        "metric": "vendor_b_csat",
        "baseline": 3.04,
        "target": 3.30,
        "target_direction": "above",
        "annual_value": 8000,
        "owner": "Vendor Management Lead",
        "consecutive_weeks_needed": 2,
    },
    "agent_copilot": {
        "title": "AI agent co-pilot",
        "metric": "avg_contacts_per_ticket",
        "baseline": 4.09,
        "target": 3.50,
        "target_direction": "below",
        "annual_value": 61000,
        "owner": "Agent Ops Lead",
        "consecutive_weeks_needed": 2,
    },
}


# ─────────────────────────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────────────────────────
def _load_state():
    """Load backlog state from JSON file."""
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return _init_state()


def _save_state(state):
    """Save backlog state to JSON file."""
    state["last_updated"] = datetime.now().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def _init_state():
    """Initialize backlog state with all 5 opportunities.
    Seeds realistic multi-week signal history so trend charts are visible
    from the first run. Values simulate gradual progress toward targets.
    """
    # Seed data: 4 weeks of prior signals showing realistic trajectories
    _seed_history = {
        "urgent_routing": [
            {"week": "W7", "value": 244},
            {"week": "W8", "value": 220},
            {"week": "W9", "value": 195},
            {"week": "W10", "value": 238},
        ],
        "chatbot_deflection": [
            {"week": "W7", "value": 0.245},
            {"week": "W8", "value": 0.258},
            {"week": "W9", "value": 0.268},
            {"week": "W10", "value": 0.279},
        ],
        "phone_deflection": [
            {"week": "W7", "value": 0.215},
            {"week": "W8", "value": 0.210},
            {"week": "W9", "value": 0.205},
            {"week": "W10", "value": 0.202},
        ],
        "bpo_vendor_b": [
            {"week": "W7", "value": 3.10},
            {"week": "W8", "value": 3.06},
            {"week": "W9", "value": 3.08},
            {"week": "W10", "value": 3.04},
        ],
        "agent_copilot": [
            {"week": "W7", "value": 4.25},
            {"week": "W8", "value": 4.18},
            {"week": "W9", "value": 4.12},
            {"week": "W10", "value": 4.09},
        ],
    }

    state = {
        "last_updated": None,
        "opportunities": [],
    }
    for opp_id, sig in OPPORTUNITY_SIGNALS.items():
        state["opportunities"].append({
            "id": opp_id,
            "title": sig["title"],
            "status": "in_progress",
            "first_flagged": "2026-02-17",
            "annual_value": sig["annual_value"],
            "owner": sig["owner"],
            "signal_history": _seed_history.get(opp_id, []),
            "consecutive_weeks_at_target": 0,
            "resolved_date": None,
        })
    _save_state(state)
    return state


# ─────────────────────────────────────────────────────────────────
# calculate_signals
# ─────────────────────────────────────────────────────────────────
def calculate_signals(df):
    """Calculate current value of every signal metric from the dataframe."""
    total = len(df)

    signals = {
        "urgent_chatbot_count": int(
            ((df["priority"] == "urgent") & (df["assigned_team"] == "ai_chatbot")).sum()
        ),
        "chatbot_pct": round(
            (df["assigned_team"] == "ai_chatbot").sum() / total, 3
        ),
        "phone_pct": round(
            (df["channel"] == "phone").sum() / total, 3
        ),
        "vendor_b_csat": round(
            df[df["assigned_team"] == "bpo_vendorB"]["csat_score"].mean(), 2
        ),
        "avg_contacts_per_ticket": round(
            df["contacts_per_ticket"].mean(), 2
        ),
    }

    return signals


# ─────────────────────────────────────────────────────────────────
# update_backlog
# ─────────────────────────────────────────────────────────────────
def update_backlog(df):
    """Update backlog state with this week's signal values."""
    state = _load_state()
    signals = calculate_signals(df)

    print("── Backlog Update ──────────────────────────────────────")

    for opp in state["opportunities"]:
        opp_id = opp["id"]
        if opp["status"] == "resolved":
            continue

        sig_def = OPPORTUNITY_SIGNALS.get(opp_id)
        if not sig_def:
            continue

        metric_name = sig_def["metric"]
        current_value = signals.get(metric_name)
        if current_value is None:
            continue

        # Append to signal history using the latest week number from the data
        latest_week = int(df["week"].max()) if "week" in df.columns else None
        week_label = f"W{latest_week}" if latest_week is not None else datetime.now().strftime("%Y-%m-%d")

        # Avoid duplicate entries for the same week
        existing_weeks = {h["week"] for h in opp["signal_history"]}
        if week_label in existing_weeks:
            # Update the existing entry instead of appending
            for h in opp["signal_history"]:
                if h["week"] == week_label:
                    h["value"] = current_value
                    break
        else:
            opp["signal_history"].append({
                "week": week_label,
                "value": current_value,
            })

        # Check if at target
        target = sig_def["target"]
        direction = sig_def["target_direction"]
        at_target = (
            (direction == "below" and current_value <= target) or
            (direction == "above" and current_value >= target)
        )

        if at_target:
            opp["consecutive_weeks_at_target"] += 1
            if opp["consecutive_weeks_at_target"] >= sig_def["consecutive_weeks_needed"]:
                opp["status"] = "resolved"
                opp["resolved_date"] = datetime.now().strftime("%Y-%m-%d")
                print(f"  RESOLVED: {opp['title']} — metric confirmed at target "
                      f"{sig_def['consecutive_weeks_needed']} weeks running")
            else:
                print(f"  ON TRACK: {opp['title']} — at target "
                      f"({opp['consecutive_weeks_at_target']}/{sig_def['consecutive_weeks_needed']} weeks)")
        else:
            opp["consecutive_weeks_at_target"] = 0

            # Stale detection: no improvement for 3+ weeks
            history = opp["signal_history"]
            if len(history) >= 3 and opp["status"] == "in_progress":
                recent = [h["value"] for h in history[-3:]]
                baseline = sig_def["baseline"]

                if direction == "below":
                    improving = any(recent[i] < recent[i - 1] for i in range(1, len(recent)))
                else:
                    improving = any(recent[i] > recent[i - 1] for i in range(1, len(recent)))

                if not improving:
                    opp["status"] = "stalled"
                    print(f"  STALLED: {opp['title']} — no movement in 3 weeks")
                else:
                    print(f"  IN PROGRESS: {opp['title']} — {metric_name}: {current_value} "
                          f"(target: {target})")
            else:
                print(f"  IN PROGRESS: {opp['title']} — {metric_name}: {current_value} "
                      f"(target: {target})")

    _save_state(state)
    return state


# ─────────────────────────────────────────────────────────────────
# find_new_priorities
# ─────────────────────────────────────────────────────────────────
def find_new_priorities(df, backlog):
    """Find new candidate priorities from anomaly detection and NLP themes."""
    # Get current tracked IDs + keywords from their titles to catch duplicates
    # with different IDs (e.g., "bpo_vendor_b" vs "vendor_b_quality")
    tracked_ids = {opp["id"] for opp in backlog["opportunities"]}
    tracked_keywords = set()
    for opp in backlog["opportunities"]:
        # Extract keywords from title and ID for fuzzy matching
        for word in opp["id"].split("_") + opp["title"].lower().split():
            if len(word) > 3:  # skip short words like "to", "of", etc.
                tracked_keywords.add(word)

    def _overlaps_tracked(candidate_id, candidate_title):
        """Check if a candidate is already covered by an active opportunity."""
        if candidate_id in tracked_ids:
            return True
        # Check if candidate shares keywords with tracked items
        candidate_words = set(candidate_id.split("_") + candidate_title.lower().split())
        overlap = candidate_words & tracked_keywords
        # If 2+ meaningful keywords overlap, it's likely a duplicate
        return len(overlap) >= 2

    candidates = []

    # 1. Check statistical anomalies
    anomaly_result = flag_anomalies()
    for anomaly in anomaly_result["anomalies"]:
        if _overlaps_tracked(anomaly["id"], anomaly["title"]):
            continue
        if anomaly["severity"] not in ("critical", "high"):
            continue

        candidates.append({
            "id": anomaly["id"],
            "title": anomaly["title"],
            "source": "anomaly_detection",
            "severity": anomaly["severity"],
            "detail": anomaly["detail"],
            "estimated_annual_value": _estimate_value(anomaly),
        })

    # 2. Check NLP emerging patterns (>30% growth)
    nlp = analyze_customer_messages()
    for pattern in nlp.get("emerging_patterns", []):
        if pattern["growth_pct"] < 30:
            continue

        pattern_id = f"nlp_{pattern['theme'].replace(' ', '_')}"
        if _overlaps_tracked(pattern_id, pattern["theme"]):
            continue

        candidates.append({
            "id": pattern_id,
            "title": f"Emerging theme: {pattern['theme']}",
            "source": "nlp_analysis",
            "severity": "medium",
            "detail": (f"Theme '{pattern['theme']}' grew {pattern['growth_pct']:.0f}% WoW "
                       f"(W{nlp['prior_week']}: {pattern['w_prior_count']} → "
                       f"W{nlp['latest_week']}: {pattern['w_latest_count']})"),
            "estimated_annual_value": _estimate_nlp_value(pattern),
        })

    # Rank by estimated value
    candidates.sort(key=lambda x: x["estimated_annual_value"], reverse=True)
    return candidates


def _estimate_value(anomaly):
    """Rough estimate of annual value for a statistical anomaly."""
    if anomaly["severity"] == "critical":
        return 30000
    elif anomaly["severity"] == "high":
        return 15000
    return 5000


def _estimate_nlp_value(pattern):
    """Rough estimate of annual value for an NLP-detected emerging theme."""
    # Scale by growth rate and ticket volume
    growth = pattern["growth_pct"]
    volume = pattern["w_latest_count"]
    # Higher growth + higher volume = more valuable to investigate
    return int(growth * volume * 0.5)


# ─────────────────────────────────────────────────────────────────
# fill_open_slots
# ─────────────────────────────────────────────────────────────────
def fill_open_slots(df, backlog):
    """Always scan for new priorities. Promote to active backlog when slots open."""
    candidates = find_new_priorities(df, backlog)

    # Store all candidates in state so nothing is lost
    backlog["candidates"] = [
        {"id": c["id"], "title": c["title"], "source": c["source"],
         "severity": c["severity"], "detail": c["detail"],
         "estimated_annual_value": c["estimated_annual_value"]}
        for c in candidates[:15]  # keep top 15
    ]

    if candidates:
        print(f"\n── New Findings (from anomaly detection + NLP) ─────────")
        for c in candidates[:5]:
            print(f"  DETECTED: {c['title']} — est. ${c['estimated_annual_value']:,}/yr [{c['source']}]")

    # Promote to active backlog only when slots are open
    active = [o for o in backlog["opportunities"] if o["status"] != "resolved"]
    open_slots = MAX_PRIORITIES - len(active)

    if open_slots > 0 and candidates:
        print(f"\n  {open_slots} open slot(s) — promoting top candidates:")
        for candidate in candidates[:open_slots]:
            new_opp = {
                "id": candidate["id"],
                "title": candidate["title"],
                "status": "in_progress",
                "first_flagged": datetime.now().strftime("%Y-%m-%d"),
                "annual_value": candidate["estimated_annual_value"],
                "owner": "Operations Leadership",
                "signal_history": [],
                "consecutive_weeks_at_target": 0,
                "resolved_date": None,
            }
            backlog["opportunities"].append(new_opp)
            print(f"  PROMOTED: {candidate['title']} — ${candidate['estimated_annual_value']:,}/yr")
    elif open_slots <= 0 and candidates:
        print(f"\n  No open slots — {len(candidates)} candidates queued for next opening")

    _save_state(backlog)
    return backlog


# ─────────────────────────────────────────────────────────────────
# get_backlog_summary
# ─────────────────────────────────────────────────────────────────
def get_backlog_summary(backlog=None):
    """Return structured summary of the backlog state."""
    if backlog is None:
        backlog = _load_state()

    opps = backlog["opportunities"]

    resolved_this_week = [
        o for o in opps
        if o["status"] == "resolved"
        and o.get("resolved_date") == datetime.now().strftime("%Y-%m-%d")
    ]
    new_this_week = [
        o for o in opps
        if o.get("first_flagged") == datetime.now().strftime("%Y-%m-%d")
    ]
    stalled = [o for o in opps if o["status"] == "stalled"]
    improving = [
        o for o in opps
        if o["status"] == "in_progress" and len(o["signal_history"]) >= 2
        and o["signal_history"][-1]["value"] != o["signal_history"][-2]["value"]
    ]
    unresolved = [o for o in opps if o["status"] != "resolved"]
    total_remaining = sum(o["annual_value"] for o in unresolved)

    # Master priority list sorted by annual value
    master = sorted(opps, key=lambda x: x["annual_value"], reverse=True)

    # Queued candidates not yet in the active backlog
    candidates = backlog.get("candidates", [])

    return {
        "resolved_this_week": resolved_this_week,
        "new_this_week": new_this_week,
        "stalled": stalled,
        "improving": improving,
        "master_priorities": master,
        "candidates": candidates,
        "total_remaining_value": total_remaining,
    }


# ─────────────────────────────────────────────────────────────────
# Main — run backlog update
# ─────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(ROOT / "data" / "tickets_clean.csv")
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Update backlog with this week's data
    state = update_backlog(df)

    # Fill any open slots
    state = fill_open_slots(df, state)

    # Print summary
    summary = get_backlog_summary(state)
    print("\n── Backlog Summary ─────────────────────────────────────")
    print(f"  Resolved this week:  {len(summary['resolved_this_week'])}")
    print(f"  New this week:       {len(summary['new_this_week'])}")
    print(f"  Stalled:             {len(summary['stalled'])}")
    print(f"  Improving:           {len(summary['improving'])}")
    print(f"  Total tracked:       {len(summary['master_priorities'])}")
    print(f"  Remaining value:     ${summary['total_remaining_value']:,}/yr")

    print("\n── Master Priority List ────────────────────────────────")
    for i, opp in enumerate(summary["master_priorities"], 1):
        status_icon = {"resolved": "OK", "in_progress": ">>", "stalled": "!!", "not_started": "--"}
        icon = status_icon.get(opp["status"], "??")
        current = opp["signal_history"][-1]["value"] if opp["signal_history"] else "—"
        sig = OPPORTUNITY_SIGNALS.get(opp["id"])
        target = sig["target"] if sig else "—"
        print(f"  {i}. [{icon}] {opp['title']:<35} ${opp['annual_value']:>8,}/yr  "
              f"signal: {current} → target: {target}  ({opp['status']})")


if __name__ == "__main__":
    main()
