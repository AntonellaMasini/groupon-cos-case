"""
Full exploratory analysis and opportunity sizing for Groupon ticket data.
Loads cleaned CSV, runs all analyses, prints results, saves to Excel.
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "tickets_clean.csv"
OUTPUT_PATH = ROOT / "output" / "analysis_results.xlsx"


def load_data():
    df = pd.read_csv(CLEAN_PATH)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


# ─────────────────────────────────────────────────────────────────
# 3a. Team Performance
# ─────────────────────────────────────────────────────────────────
def team_performance(df):
    print("\n══ 3a. TEAM PERFORMANCE ════════════════════════════════")
    g = df.groupby("assigned_team")
    result = pd.DataFrame({
        "ticket_count": g.size(),
        "avg_cost": g["cost_usd"].mean(),
        "avg_csat": g["csat_score"].mean(),
        "resolution_rate": g["is_resolved"].mean() * 100,
        "escalation_rate": g["is_escalated"].mean() * 100,
        "abandonment_rate": g["is_abandoned"].mean() * 100,
        "avg_contacts": g["contacts_per_ticket"].mean(),
    }).round(2)
    result = result.sort_values("ticket_count", ascending=False)
    print(result.to_string())
    return result


# ─────────────────────────────────────────────────────────────────
# 3b. Channel Performance
# ─────────────────────────────────────────────────────────────────
def channel_performance(df):
    print("\n══ 3b. CHANNEL PERFORMANCE ═════════════════════════════")
    g = df.groupby("channel")
    result = pd.DataFrame({
        "ticket_count": g.size(),
        "avg_cost": g["cost_usd"].mean(),
        "avg_csat": g["csat_score"].mean(),
        "avg_first_response_min": g["first_response_min"].mean(),
        "avg_resolution_min": g["resolution_min"].mean(),
    }).round(2)
    result = result.sort_values("ticket_count", ascending=False)
    print(result.to_string())
    return result


# ─────────────────────────────────────────────────────────────────
# 3c. Category Analysis
# ─────────────────────────────────────────────────────────────────
def category_analysis(df):
    print("\n══ 3c. CATEGORY ANALYSIS ═══════════════════════════════")
    g = df.groupby("category")
    result = pd.DataFrame({
        "ticket_count": g.size(),
        "avg_cost": g["cost_usd"].mean(),
        "escalation_rate": g["is_escalated"].mean() * 100,
        "avg_contacts": g["contacts_per_ticket"].mean(),
        "avg_csat": g["csat_score"].mean(),
    }).round(2)
    result = result.sort_values("ticket_count", ascending=False)
    print(result.to_string())
    return result


# ─────────────────────────────────────────────────────────────────
# 3d. Weekly Trends W7-W10 with WoW Deltas
# ─────────────────────────────────────────────────────────────────
def weekly_trends(df):
    print("\n══ 3d. WEEKLY TRENDS (W7-W10) ══════════════════════════")
    g = df.groupby("week")
    result = pd.DataFrame({
        "ticket_count": g.size(),
        "total_cost": g["cost_usd"].sum(),
        "avg_cost": g["cost_usd"].mean(),
        "avg_csat": g["csat_score"].mean(),
        "resolution_rate": g["is_resolved"].mean() * 100,
        "escalation_rate": g["is_escalated"].mean() * 100,
    }).round(2)
    print(result.to_string())

    # Week-over-week deltas
    print("\n── Week-over-Week Deltas ───────────────────────────────")
    metrics = ["ticket_count", "total_cost", "avg_cost", "avg_csat",
               "resolution_rate", "escalation_rate"]
    weeks = sorted(result.index)
    for i in range(1, len(weeks)):
        w_prev, w_curr = weeks[i - 1], weeks[i]
        print(f"\n  W{w_curr} vs W{w_prev}:")
        for m in metrics:
            prev_val = result.loc[w_prev, m]
            curr_val = result.loc[w_curr, m]
            delta = curr_val - prev_val
            if prev_val != 0:
                pct = (delta / abs(prev_val)) * 100
            else:
                pct = 0.0
            if m in ("avg_cost", "escalation_rate"):
                direction = "improving" if delta < -0.5 else ("worsening" if delta > 0.5 else "flat")
            elif m in ("avg_csat", "resolution_rate"):
                direction = "improving" if delta > 0.5 else ("worsening" if delta < -0.5 else "flat")
            else:
                direction = "increasing" if delta > 0 else ("decreasing" if delta < 0 else "flat")
            print(f"    {m:<20} {prev_val:>10.2f} → {curr_val:>10.2f}  "
                  f"({pct:+.1f}%) {direction}")

    return result


# ─────────────────────────────────────────────────────────────────
# 3e. Key Anomalies
# ─────────────────────────────────────────────────────────────────
def key_anomalies(df):
    print("\n══ 3e. KEY ANOMALIES ═══════════════════════════════════")

    # ANOMALY 1: Urgent tickets routed to chatbot
    urgent = df[df["priority"] == "urgent"]
    urgent_chatbot = urgent[urgent["assigned_team"] == "ai_chatbot"]
    urgent_inhouse = urgent[urgent["assigned_team"] == "in_house"]
    a1_count = len(urgent_chatbot)
    a1_chatbot_res = urgent_chatbot["is_resolved"].mean() * 100
    a1_inhouse_res = urgent_inhouse["is_resolved"].mean() * 100
    a1_chatbot_csat = urgent_chatbot["csat_score"].mean()
    a1_inhouse_csat = urgent_inhouse["csat_score"].mean()
    a1_csat_gap = a1_inhouse_csat - a1_chatbot_csat

    print(f"\n  ANOMALY 1: Urgent tickets routed to chatbot")
    print(f"    Count: {a1_count}")
    print(f"    Chatbot resolution rate for urgent: {a1_chatbot_res:.1f}%")
    print(f"    In-house resolution rate for urgent: {a1_inhouse_res:.1f}%")
    print(f"    CSAT gap: {a1_csat_gap:.2f} pts")

    # ANOMALY 2: Chatbot containment failure
    chatbot = df[df["assigned_team"] == "ai_chatbot"]
    chatbot_esc_rate = chatbot["is_escalated"].mean() * 100
    benchmark = 15.0
    excess_per_4wk = int(len(chatbot) * (chatbot_esc_rate / 100 - benchmark / 100))

    print(f"\n  ANOMALY 2: Chatbot containment failure")
    print(f"    Chatbot escalation rate: {chatbot_esc_rate:.1f}%")
    print(f"    Industry benchmark: ~{benchmark:.0f}%")
    print(f"    Excess escalations per 4 weeks: {excess_per_4wk}")

    # ANOMALY 3: BPO Vendor B quality gap
    teams_csat = df.groupby("assigned_team")["csat_score"].mean()
    teams_contacts = df.groupby("assigned_team")["contacts_per_ticket"].mean()
    vb_csat = teams_csat.get("bpo_vendorB", np.nan)
    va_csat = teams_csat.get("bpo_vendorA", np.nan)
    ih_csat = teams_csat.get("in_house", np.nan)
    vb_contacts = teams_contacts.get("bpo_vendorB", np.nan)
    va_contacts = teams_contacts.get("bpo_vendorA", np.nan)

    print(f"\n  ANOMALY 3: BPO Vendor B quality gap")
    print(f"    Vendor B CSAT: {vb_csat:.2f} vs Vendor A: {va_csat:.2f} vs In-house: {ih_csat:.2f}")
    print(f"    Vendor B contacts/ticket: {vb_contacts:.1f} vs Vendor A: {va_contacts:.1f}")

    # ANOMALY 4: Phone channel cost
    channel_stats = df.groupby("channel").agg(
        avg_cost=("cost_usd", "mean"),
        avg_csat=("csat_score", "mean")
    )
    phone_cost = channel_stats.loc["phone", "avg_cost"]
    chat_cost = channel_stats.loc["chat", "avg_cost"]
    phone_csat = channel_stats.loc["phone", "avg_csat"]
    chat_csat = channel_stats.loc["chat", "avg_csat"]
    ratio = phone_cost / chat_cost

    print(f"\n  ANOMALY 4: Phone channel cost")
    print(f"    Phone: ${phone_cost:.2f}/ticket vs Chat: ${chat_cost:.2f}/ticket ({ratio:.1f}x gap)")
    print(f"    Phone CSAT: {phone_csat:.2f} vs Chat CSAT: {chat_csat:.2f}")

    # Build summary dataframe for Excel
    anomalies_data = [
        {"anomaly": "Urgent tickets → chatbot", "detail": f"{a1_count} tickets",
         "impact": f"Resolution {a1_chatbot_res:.1f}% vs {a1_inhouse_res:.1f}%, CSAT gap {a1_csat_gap:.2f}"},
        {"anomaly": "Chatbot containment failure", "detail": f"Escalation rate {chatbot_esc_rate:.1f}%",
         "impact": f"{excess_per_4wk} excess escalations vs {benchmark:.0f}% benchmark"},
        {"anomaly": "BPO Vendor B quality", "detail": f"CSAT {vb_csat:.2f}",
         "impact": f"vs Vendor A {va_csat:.2f}, In-house {ih_csat:.2f}"},
        {"anomaly": "Phone channel cost", "detail": f"${phone_cost:.2f}/ticket",
         "impact": f"{ratio:.1f}x more than chat (${chat_cost:.2f}) with similar CSAT"},
    ]
    return pd.DataFrame(anomalies_data)


# ─────────────────────────────────────────────────────────────────
# 3f. Opportunity Sizing
# ─────────────────────────────────────────────────────────────────
def opportunity_sizing(df):
    """Size all opportunities using the canonical formulas from agent/tools.py."""
    print("\n══ 3f. OPPORTUNITY SIZING ══════════════════════════════")

    # Import from the single source of truth to avoid duplicate formulas
    from agent.tools import size_opportunity

    opp_names = [
        "chatbot_deflection", "agent_copilot", "urgent_routing",
        "phone_deflection", "bpo_vendor_b",
    ]

    opps = []
    total = 0
    for name in opp_names:
        result = size_opportunity(name)
        label = result["opportunity"]
        savings = result["annual_savings"]
        range_low = result.get("range_low", "")
        range_high = result.get("range_high", "")
        print(f"  {label:<50} ${savings:>10,}/yr  (range: ${range_low:,}–${range_high:,})")
        opps.append({"opportunity": label, "annual_savings_usd": savings})
        total += savings

    print(f"  {'─' * 62}")
    print(f"  {'TOTAL ANNUAL OPPORTUNITY':<50} ${total:>10,}/yr")

    opp_df = pd.DataFrame(opps)
    opp_df.loc[len(opp_df)] = {"opportunity": "TOTAL", "annual_savings_usd": round(total)}
    return opp_df


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    print(f"Loaded {len(df):,} clean tickets (W7-W10)")

    team_df = team_performance(df)
    channel_df = channel_performance(df)
    category_df = category_analysis(df)
    trends_df = weekly_trends(df)
    anomalies_df = key_anomalies(df)
    opps_df = opportunity_sizing(df)

    # Save to Excel
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        team_df.to_excel(writer, sheet_name="Team Performance")
        channel_df.to_excel(writer, sheet_name="Channel Performance")
        category_df.to_excel(writer, sheet_name="Category Analysis")
        trends_df.to_excel(writer, sheet_name="Weekly Trends")
        anomalies_df.to_excel(writer, sheet_name="Anomalies", index=False)
        opps_df.to_excel(writer, sheet_name="Opportunity Sizing", index=False)

    print(f"\n  All results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
