"""
Slack notification for the weekly Ops Intelligence Brief.
Sends a condensed version of the brief via Incoming Webhook.
Contains all 4 required elements: top 5 issues, WoW comparison,
actions with owners, and watch list.
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def build_message(backlog_summary, weekly_stats, watch_list, brief_url="http://localhost:8501"):
    """Build the Slack message payload from pipeline outputs."""

    week_num = weekly_stats.get("current_week_num", "?")
    comparison = weekly_stats.get("comparison", {})

    blocks = []

    # ── Header ────────────────────────────────────────────────
    blocks.append({
        "type": "header",
        "text": {"type": "plain_text", "text": f"Groupon Ops Intelligence Brief — Week {week_num}"}
    })
    blocks.append({"type": "divider"})

    # ── Resolved this week ────────────────────────────────────
    resolved = backlog_summary.get("resolved_this_week", [])
    if resolved:
        resolved_lines = [":white_check_mark: *RESOLVED THIS WEEK*"]
        for r in resolved:
            resolved_lines.append(f":white_check_mark: {r['title']} — metric confirmed at target 2 weeks running")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(resolved_lines)}
        })

    # ── New this week ─────────────────────────────────────────
    new = backlog_summary.get("new_this_week", [])
    if new:
        new_lines = [":zap: *NEW THIS WEEK*"]
        for n in new:
            new_lines.append(f":zap: {n['title']} — ${n['annual_value']:,}/yr")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(new_lines)}
        })

    # ── Stalled ───────────────────────────────────────────────
    stalled = backlog_summary.get("stalled", [])
    if stalled:
        stalled_lines = [":warning: *STALLED — NO MOVEMENT*"]
        for s in stalled:
            weeks_tracked = len(s.get("signal_history", []))
            stalled_lines.append(f":no_entry: {s['title']} — {weeks_tracked} weeks, recommend escalation")
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(stalled_lines)}
        })

    # ── Top 5 this week ───────────────────────────────────────
    master = backlog_summary.get("master_priorities", [])
    top5_lines = [":bar_chart: *TOP 5 THIS WEEK*"]

    status_emoji = {
        "not_started": ":red_circle:",
        "in_progress": ":large_yellow_circle:",
        "resolved": ":large_green_circle:",
        "stalled": ":no_entry:",
    }
    trend_arrow = {
        "improving": ":chart_with_upwards_trend:",
        "worsening": ":chart_with_downwards_trend:",
        "flat": ":arrow_right:",
    }

    from agent.backlog import OPPORTUNITY_SIGNALS

    for i, opp in enumerate(master[:5], 1):
        emoji = status_emoji.get(opp["status"], ":white_circle:")
        sig_def = OPPORTUNITY_SIGNALS.get(opp["id"])
        current = opp["signal_history"][-1]["value"] if opp.get("signal_history") else "—"
        target = sig_def["target"] if sig_def else "—"

        top5_lines.append(
            f"{i}. {emoji} *{opp['title']}* — ${opp['annual_value']:,}/yr\n"
            f"      Signal: `{current}` → target `{target}`\n"
            f"      Owner: {opp['owner']}"
        )

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(top5_lines)}
    })
    blocks.append({"type": "divider"})

    # ── Week over week ────────────────────────────────────────
    wow_metrics = {
        "ticket_volume": "Volume",
        "avg_csat": "CSAT",
        "resolution_rate": "Resolution rate",
        "escalation_rate": "Escalation rate",
        "avg_cost_per_ticket": "Avg cost",
    }

    wow_lines = [":calendar: *WEEK OVER WEEK*"]
    for key, label in wow_metrics.items():
        data = comparison.get(key, {})
        prev = data.get("prior_week", "—")
        curr = data.get("current_week", "—")
        pct = data.get("pct_change", 0)
        direction = data.get("direction", "flat")
        arrow = trend_arrow.get(direction, ":arrow_right:")

        if "cost" in key:
            wow_lines.append(f"{arrow} {label}: `${prev}` → `${curr}` ({pct:+.1f}%)")
        elif "rate" in key:
            wow_lines.append(f"{arrow} {label}: `{prev}%` → `{curr}%` ({pct:+.1f}%)")
        else:
            wow_lines.append(f"{arrow} {label}: `{prev}` → `{curr}` ({pct:+.1f}%)")

    blocks.append({
        "type": "section",
        "text": {"type": "mrkdwn", "text": "\n".join(wow_lines)}
    })
    blocks.append({"type": "divider"})

    # ── Watch list ────────────────────────────────────────────
    if watch_list:
        watch_lines = [":eyes: *WATCH LIST*"]
        for item in watch_list[:5]:
            if "theme" in item:
                watch_lines.append(
                    f":warning: *{item['theme']}* — grew {item['growth_pct']:.0f}% WoW "
                    f"({item['w_prior_count']} → {item['w_latest_count']} mentions)"
                )
            elif "metric" in item:
                watch_lines.append(
                    f":warning: *{item['metric']}* — moved {item['pct_change']:+.1f}%"
                )
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "\n".join(watch_lines)}
        })
        blocks.append({"type": "divider"})

    # ── Footer ────────────────────────────────────────────────
    total_remaining = backlog_summary.get("total_remaining_value", 0)
    blocks.append({
        "type": "context",
        "elements": [{
            "type": "mrkdwn",
            "text": (f"Total remaining opportunity: *${total_remaining:,}/yr* | "
                     f"<{brief_url}|Full brief in Streamlit>")
        }]
    })

    return {"blocks": blocks}


def send_notification(message):
    """Post the message to Slack via webhook."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url or not webhook_url.startswith("https://"):
        print("  WARNING: SLACK_WEBHOOK_URL not configured — skipping Slack notification")
        print("  (Set a valid webhook URL in .env to enable Slack delivery)")
        return False

    response = requests.post(
        webhook_url,
        json=message,
        headers={"Content-Type": "application/json"},
    )

    if response.status_code == 200:
        print("  Slack notification sent successfully")
        return True
    else:
        print(f"  Slack notification failed: {response.status_code} — {response.text}")
        return False


def build_watch_list(nlp_result, weekly_trends):
    """Combine NLP emerging patterns and flagged metrics into a watch list."""
    watch_list = []

    # NLP emerging patterns
    for pattern in nlp_result.get("emerging_patterns", [])[:5]:
        watch_list.append({
            "type": "nlp",
            "theme": pattern["theme"],
            "growth_pct": pattern["growth_pct"],
            "w_prior_count": pattern["w_prior_count"],
            "w_latest_count": pattern["w_latest_count"],
        })

    # Flagged metrics (>5% wrong direction)
    comparison = weekly_trends.get("comparison", {})
    for key in weekly_trends.get("flagged_metrics", []):
        data = comparison.get(key, {})
        watch_list.append({
            "type": "metric",
            "metric": key,
            "pct_change": data.get("pct_change", 0),
            "direction": data.get("direction", "unknown"),
        })

    return watch_list


def main():
    """Run Slack notification with current pipeline data."""
    from agent.tools import get_weekly_trends, analyze_customer_messages
    from agent.backlog import get_backlog_summary

    print("── Slack Notification ──────────────────────────────────")

    weekly_stats = get_weekly_trends()
    nlp = analyze_customer_messages()
    backlog_summary = get_backlog_summary()
    watch_list = build_watch_list(nlp, weekly_stats)

    message = build_message(backlog_summary, weekly_stats, watch_list)
    send_notification(message)

    # Also save message to file for debugging
    output_path = ROOT / "output" / "slack_message.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(message, indent=2))
    print(f"  Message saved to {output_path}")


if __name__ == "__main__":
    main()
