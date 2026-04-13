"""
Agent tools for the Groupon Ops Intelligence pipeline.
Each tool is a callable function + an Anthropic tool schema.
Tools operate on the cleaned ticket CSV data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "tickets_clean.csv"

# Module-level data cache to avoid re-reading CSV on every tool call
_df_cache = None


# ─────────────────────────────────────────────────────────────────
# Opportunity metadata: owners, actions, timelines, root causes
# ─────────────────────────────────────────────────────────────────
OPPORTUNITY_META = {
    "chatbot_deflection": {
        "owner": "AI Product Lead",
        "action": "Retrain chatbot on top 3 deflection-ready categories (order_status, account, voucher_problem). Add intent classification layer to route simple tickets to chatbot before BPO queue.",
        "timeline": "SHORT TERM (this month)",
        "kpi": "Chatbot deflection rate from 27.9% to 43%",
        "root_cause": "Chatbot is only trained on a narrow set of intents — high-volume simple categories (order_status, account) still default to BPO queues instead of chatbot.",
    },
    "agent_copilot": {
        "owner": "Agent Ops Lead",
        "action": "Deploy AI co-pilot for in-house and BPO agents: auto-suggest responses, surface customer history, pre-fill resolution templates. Pilot with top 10 agents first.",
        "timeline": "SHORT TERM (this month)",
        "kpi": "Avg contacts per ticket from 4.1 to 3.5",
        "root_cause": "Agents lack real-time context — no auto-surfaced customer history or suggested responses, leading to extra back-and-forth contacts per ticket.",
    },
    "urgent_routing": {
        "owner": "Head of CX Ops",
        "action": "Add priority gate in routing engine: block chatbot assignment for urgent/high tickets. Route urgent tickets directly to in-house senior agents.",
        "timeline": "IMMEDIATE (this week)",
        "kpi": "Urgent tickets routed to chatbot from 238 to <10",
        "root_cause": "Routing engine has no priority gating rule — urgent and high-priority tickets are assigned to AI chatbot identically to low-priority ones.",
    },
    "phone_deflection": {
        "owner": "Channel Strategy Lead",
        "action": "Add IVR prompt offering chat/callback option before phone queue. Launch SMS deflection for order_status and refund categories.",
        "timeline": "MEDIUM TERM (this quarter)",
        "kpi": "Phone ticket share from 20.1% to 16.1%",
        "root_cause": "No IVR deflection layer — customers calling for simple issues (order status, refunds) are not offered a chat or callback alternative before entering the phone queue.",
    },
    "bpo_vendor_b": {
        "owner": "Vendor Management Lead",
        "action": "Conduct QA audit of Vendor B bottom-10 agents. Implement mandatory retraining on escalation handling. Set 4-week improvement target or trigger contract review.",
        "timeline": "IMMEDIATE (this week)",
        "kpi": "Vendor B CSAT from 3.04 to 3.30",
        "root_cause": "Vendor B agents have higher contacts-per-ticket (5.4 vs 4.8) and lower resolution quality, suggesting insufficient training on complex ticket categories.",
    },
}

# Map opportunity names to their IDs for lookup
OPPORTUNITY_ID_MAP = {
    "Expand chatbot deflection": "chatbot_deflection",
    "AI agent co-pilot": "agent_copilot",
    "Fix urgent ticket routing": "urgent_routing",
    "Phone to chat deflection": "phone_deflection",
    "BPO Vendor B quality intervention": "bpo_vendor_b",
}


def _load_data():
    """Load cleaned ticket data, cached to avoid repeated disk reads."""
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(CLEAN_PATH)
        _df_cache["created_at"] = pd.to_datetime(_df_cache["created_at"])
    return _df_cache.copy()


def invalidate_cache():
    """Clear the data cache (call after new data is cleaned)."""
    global _df_cache
    _df_cache = None


# ─────────────────────────────────────────────────────────────────
# Tool 1: check_data_quality
# ─────────────────────────────────────────────────────────────────
def check_data_quality(filepath=None):
    """Load CSV and return data quality summary."""
    df = _load_data()
    total = len(df)

    missing = {col: int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0}
    missing_pct = {col: round(count / total * 100, 1) for col, count in missing.items()}

    urgent_chatbot = int(((df["priority"] == "urgent") & (df["assigned_team"] == "ai_chatbot")).sum())
    missing_csat_rate = round(df["csat_score"].isnull().sum() / total * 100, 1)
    high_touch = int((df["contacts_per_ticket"] > 5).sum())

    anomalies = []
    if urgent_chatbot > 50:
        anomalies.append(f"CRITICAL: {urgent_chatbot} urgent tickets routed to AI chatbot")
    if missing_csat_rate > 20:
        anomalies.append(f"WARNING: {missing_csat_rate}% of tickets missing CSAT scores")
    if high_touch > total * 0.15:
        anomalies.append(f"WARNING: {high_touch} high-touch tickets (>5 contacts) — {round(high_touch/total*100,1)}%")

    return {
        "row_count": total,
        "columns": list(df.columns),
        "weeks": sorted(df["week"].unique().tolist()),
        "missing_values": missing,
        "missing_pct": missing_pct,
        "anomalies_found": anomalies,
        "urgent_chatbot_count": urgent_chatbot,
        "missing_csat_rate": missing_csat_rate,
        "high_touch_count": high_touch,
    }


# ─────────────────────────────────────────────────────────────────
# Tool 2: analyze_metric
# ─────────────────────────────────────────────────────────────────
def analyze_metric(metric, group_by):
    """Group data by a dimension and compute summary stats for a metric."""
    df = _load_data()

    valid_metrics = ["cost_usd", "csat_score", "resolution_min", "first_response_min",
                     "contacts_per_ticket", "is_resolved", "is_escalated", "is_abandoned"]
    valid_groups = ["assigned_team", "channel", "category", "priority", "market", "week"]

    if metric not in valid_metrics:
        return {"error": f"Invalid metric '{metric}'. Valid: {valid_metrics}"}
    if group_by not in valid_groups:
        return {"error": f"Invalid group_by '{group_by}'. Valid: {valid_groups}"}

    g = df.groupby(group_by)[metric]
    result = pd.DataFrame({
        "count": g.count(),
        "mean": g.mean().round(3),
        "median": g.median().round(3),
        "std": g.std().round(3),
        "min": g.min().round(3),
        "max": g.max().round(3),
    })

    # For boolean metrics, show as percentages
    if metric.startswith("is_"):
        result["mean"] = (result["mean"] * 100).round(1)
        result["median"] = (result["median"] * 100).round(1)

    result = result.sort_values("mean", ascending=False)
    return {
        "metric": metric,
        "group_by": group_by,
        "summary": result.reset_index().to_dict(orient="records"),
    }


# ─────────────────────────────────────────────────────────────────
# Tool 3: flag_anomalies
# ─────────────────────────────────────────────────────────────────
def flag_anomalies():
    """Run anomaly detection across all key metrics."""
    df = _load_data()
    anomalies = []

    # 1. Urgent tickets routed to chatbot
    urgent_chatbot = df[(df["priority"] == "urgent") & (df["assigned_team"] == "ai_chatbot")]
    urgent_inhouse = df[(df["priority"] == "urgent") & (df["assigned_team"] == "in_house")]
    if len(urgent_chatbot) > 50:
        anomalies.append({
            "id": "urgent_routing",
            "severity": "critical",
            "title": "Urgent tickets misrouted to AI chatbot",
            "detail": (f"{len(urgent_chatbot)} urgent tickets sent to chatbot. "
                       f"Chatbot resolution: {urgent_chatbot['is_resolved'].mean()*100:.1f}% "
                       f"vs in-house: {urgent_inhouse['is_resolved'].mean()*100:.1f}%. "
                       f"CSAT gap: {urgent_inhouse['csat_score'].mean() - urgent_chatbot['csat_score'].mean():.2f} pts."),
        })

    # 2. Chatbot containment failure
    chatbot = df[df["assigned_team"] == "ai_chatbot"]
    chatbot_esc = chatbot["is_escalated"].mean() * 100
    if chatbot_esc > 20:
        benchmark = 15.0
        excess = int(len(chatbot) * (chatbot_esc / 100 - benchmark / 100))
        anomalies.append({
            "id": "chatbot_containment",
            "severity": "high",
            "title": "Chatbot escalation rate exceeds benchmark",
            "detail": (f"Chatbot escalation rate: {chatbot_esc:.1f}% vs industry benchmark ~{benchmark:.0f}%. "
                       f"{excess} excess escalations per 4 weeks."),
        })

    # 3. BPO Vendor B quality
    vendor_b = df[df["assigned_team"] == "bpo_vendorB"]
    vendor_a = df[df["assigned_team"] == "bpo_vendorA"]
    inhouse = df[df["assigned_team"] == "in_house"]
    vb_csat = vendor_b["csat_score"].mean()
    if vb_csat < 3.1:
        anomalies.append({
            "id": "vendor_b_quality",
            "severity": "high",
            "title": "BPO Vendor B quality gap",
            "detail": (f"Vendor B CSAT: {vb_csat:.2f} vs Vendor A: {vendor_a['csat_score'].mean():.2f} "
                       f"vs in-house: {inhouse['csat_score'].mean():.2f}. "
                       f"Vendor B contacts/ticket: {vendor_b['contacts_per_ticket'].mean():.1f} "
                       f"vs Vendor A: {vendor_a['contacts_per_ticket'].mean():.1f}."),
        })

    # 4. Phone channel cost
    phone = df[df["channel"] == "phone"]
    chat = df[df["channel"] == "chat"]
    phone_cost = phone["cost_usd"].mean()
    chat_cost = chat["cost_usd"].mean()
    ratio = phone_cost / chat_cost
    if ratio > 2.5:
        anomalies.append({
            "id": "phone_cost",
            "severity": "high",
            "title": "Phone channel cost premium",
            "detail": (f"Phone: ${phone_cost:.2f}/ticket vs chat: ${chat_cost:.2f}/ticket ({ratio:.1f}x gap). "
                       f"Phone CSAT: {phone['csat_score'].mean():.2f} vs chat: {chat['csat_score'].mean():.2f} "
                       f"— similar satisfaction at very different cost."),
        })

    # 5. High-touch ticket concentration
    high_touch = df[df["is_high_touch"]]
    ht_pct = len(high_touch) / len(df) * 100
    if ht_pct > 15:
        anomalies.append({
            "id": "high_touch",
            "severity": "medium",
            "title": "High-touch ticket concentration",
            "detail": (f"{len(high_touch)} tickets ({ht_pct:.1f}%) require >5 contacts. "
                       f"Avg cost: ${high_touch['cost_usd'].mean():.2f} vs "
                       f"${df[~df['is_high_touch']]['cost_usd'].mean():.2f} for normal tickets."),
        })

    # ── Statistical anomaly detection ────────────────────────────
    # Scan all metric × dimension combinations for z-score outliers.
    # This catches anything the hardcoded checks above might miss.
    known_ids = {a["id"] for a in anomalies}
    metrics = ["cost_usd", "csat_score", "contacts_per_ticket", "resolution_min", "first_response_min"]
    dimensions = ["assigned_team", "channel", "category", "market", "priority"]

    for metric in metrics:
        for dim in dimensions:
            groups = df.groupby(dim)[metric]
            means = groups.mean()
            if len(means) < 3:
                continue
            overall_mean = means.mean()
            overall_std = means.std()
            if overall_std == 0:
                continue

            for group_name, group_mean in means.items():
                z = abs(group_mean - overall_mean) / overall_std
                if z < 2.0:
                    continue

                # Build a unique ID to avoid duplicates with hardcoded checks
                anomaly_id = f"stat_{dim}_{group_name}_{metric}".lower().replace(" ", "_")
                if anomaly_id in known_ids:
                    continue

                direction = "above" if group_mean > overall_mean else "below"
                group_count = groups.get_group(group_name).count()

                # Determine severity by z-score
                severity = "high" if z >= 3.0 else "medium"

                anomalies.append({
                    "id": anomaly_id,
                    "severity": severity,
                    "title": f"Statistical outlier: {group_name} {metric} ({dim})",
                    "detail": (f"{group_name} has {metric} = {group_mean:.2f}, "
                               f"which is {z:.1f} std devs {direction} the mean of {overall_mean:.2f} "
                               f"across {dim}. Based on {group_count} observations."),
                })
                known_ids.add(anomaly_id)

    # Also check for WoW spikes in the latest week per dimension
    weeks = sorted(df["week"].unique())
    if len(weeks) >= 2:
        latest, prior = weeks[-1], weeks[-2]
        df_latest = df[df["week"] == latest]
        df_prior = df[df["week"] == prior]

        for dim in dimensions:
            for metric in metrics:
                latest_means = df_latest.groupby(dim)[metric].mean()
                prior_means = df_prior.groupby(dim)[metric].mean()

                for group_name in latest_means.index:
                    if group_name not in prior_means.index:
                        continue
                    curr_val = latest_means[group_name]
                    prev_val = prior_means[group_name]
                    if prev_val == 0:
                        continue
                    pct_change = (curr_val - prev_val) / abs(prev_val) * 100

                    if abs(pct_change) < 15:  # only flag big weekly swings
                        continue

                    anomaly_id = f"wow_{dim}_{group_name}_{metric}".lower().replace(" ", "_")
                    if anomaly_id in known_ids:
                        continue

                    direction = "increased" if pct_change > 0 else "decreased"
                    severity = "high" if abs(pct_change) >= 25 else "medium"

                    anomalies.append({
                        "id": anomaly_id,
                        "severity": severity,
                        "title": f"WoW spike: {group_name} {metric} ({dim})",
                        "detail": (f"{group_name} {metric} {direction} {abs(pct_change):.1f}% WoW "
                                   f"(W{prior}: {prev_val:.2f} → W{latest}: {curr_val:.2f})."),
                    })
                    known_ids.add(anomaly_id)

    return {"anomalies": anomalies, "count": len(anomalies)}


# ─────────────────────────────────────────────────────────────────
# Tool 4: size_opportunity
# ─────────────────────────────────────────────────────────────────
def size_opportunity(opportunity_name):
    """Calculate annual savings for a named opportunity."""
    df = _load_data()
    total = len(df)
    annual_factor = 13  # 52 weeks / 4 weeks

    valid = ["chatbot_deflection", "agent_copilot", "urgent_routing", "phone_deflection", "bpo_vendor_b"]
    if opportunity_name not in valid:
        return {"error": f"Invalid opportunity. Valid: {valid}"}

    if opportunity_name == "chatbot_deflection":
        chatbot = df[df["assigned_team"] == "ai_chatbot"]
        chatbot_pct = len(chatbot) / total
        chatbot_cost = chatbot["cost_usd"].mean()
        bpo = df[df["assigned_team"].isin(["bpo_vendorA", "bpo_vendorB"])]
        bpo_cost = bpo["cost_usd"].mean()
        target_pct = 0.43
        additional = (target_pct - chatbot_pct) * total
        saving = additional * (bpo_cost - chatbot_cost) * annual_factor
        base = round(saving)
        return {
            "opportunity": "Expand chatbot deflection",
            "current": f"{chatbot_pct*100:.1f}%",
            "target": f"{target_pct*100:.0f}%",
            "additional_tickets_deflected_4wk": int(additional),
            "saving_per_ticket": round(bpo_cost - chatbot_cost, 2),
            "annual_savings": base,
            "range_low": round(base * 0.75),
            "range_high": round(base * 1.25),
            "confidence": "Medium — assumes BPO cost structure holds at higher chatbot volume and chatbot quality does not degrade with expanded scope",
            "assumptions": "Deflected tickets come from BPO queues. Cost saving = BPO avg - chatbot avg.",
        }

    elif opportunity_name == "agent_copilot":
        avg_contacts = df["contacts_per_ticket"].mean()
        avg_cost = df["cost_usd"].mean()
        target = 3.5
        reduction = avg_contacts - target
        cost_per_contact = avg_cost / avg_contacts if avg_contacts > 0 else 0
        saving = reduction * cost_per_contact * total * annual_factor
        base = round(saving)
        return {
            "opportunity": "AI agent co-pilot",
            "current_contacts": round(avg_contacts, 2),
            "target_contacts": target,
            "cost_per_contact": round(cost_per_contact, 2),
            "annual_savings": base,
            "range_low": round(base * 0.7),
            "range_high": round(base * 1.3),
            "confidence": "Medium — assumes co-pilot adoption is consistent across agent tiers and cost scales linearly with contact reduction",
            "assumptions": "Co-pilot reduces avg contacts from current to 3.5. Cost scales linearly with contacts.",
        }

    elif opportunity_name == "urgent_routing":
        urgent_chatbot = df[(df["priority"] == "urgent") & (df["assigned_team"] == "ai_chatbot")]
        count = len(urgent_chatbot)
        esc_rate = urgent_chatbot["is_escalated"].mean()
        abn_rate = urgent_chatbot["is_abandoned"].mean()
        res_rate = urgent_chatbot["is_resolved"].mean()
        non_chatbot = df[df["assigned_team"] != "ai_chatbot"]
        human_cost = non_chatbot["cost_usd"].mean()
        rework = count * esc_rate * human_cost
        abandonment = count * abn_rate * 100  # $100 avg urgent order value
        repeat = count * res_rate * 0.30 * human_cost
        saving = (rework + abandonment + repeat) * annual_factor
        base = round(saving)
        return {
            "opportunity": "Fix urgent ticket routing",
            "misrouted_tickets_4wk": count,
            "chatbot_escalation_rate": f"{esc_rate*100:.1f}%",
            "chatbot_abandonment_rate": f"{abn_rate*100:.1f}%",
            "annual_savings": base,
            "range_low": round(base * 0.8),
            "range_high": round(base * 1.2),
            "confidence": "High — routing fix is deterministic; savings depend on escalation and abandonment rates which are directly observed",
            "assumptions": ("Rework cost for escalated tickets + lost revenue from abandoned urgent tickets "
                            "($100 avg order value) + repeat contacts for 30% of poorly resolved tickets."),
        }

    elif opportunity_name == "phone_deflection":
        phone = df[df["channel"] == "phone"]
        chat = df[df["channel"] == "chat"]
        deflection_rate = 0.20
        deflected = len(phone) * deflection_rate
        saving = deflected * (phone["cost_usd"].mean() - chat["cost_usd"].mean()) * annual_factor
        base = round(saving)
        return {
            "opportunity": "Phone to chat deflection",
            "phone_tickets_4wk": len(phone),
            "deflection_target": "20%",
            "phone_cost": round(phone["cost_usd"].mean(), 2),
            "chat_cost": round(chat["cost_usd"].mean(), 2),
            "annual_savings": base,
            "range_low": round(base * 0.8),
            "range_high": round(base * 1.2),
            "confidence": "Medium — 20% deflection is conservative; actual depends on IVR acceptance rate and customer willingness to switch channels",
            "assumptions": "20% of phone volume migrated to chat. CSAT is similar across both channels.",
        }

    elif opportunity_name == "bpo_vendor_b":
        vb = df[df["assigned_team"] == "bpo_vendorB"]
        va = df[df["assigned_team"] == "bpo_vendorA"]
        contact_saving = vb["contacts_per_ticket"].mean() - va["contacts_per_ticket"].mean()
        vb_contacts = vb["contacts_per_ticket"].mean()
        cost_per_contact = vb["cost_usd"].mean() / vb_contacts if vb_contacts > 0 else 0
        saving = contact_saving * cost_per_contact * len(vb) * annual_factor
        base = round(saving)
        return {
            "opportunity": "BPO Vendor B quality intervention",
            "vendor_b_contacts": round(vb_contacts, 2),
            "vendor_a_contacts": round(va["contacts_per_ticket"].mean(), 2),
            "vendor_b_csat": round(vb["csat_score"].mean(), 2),
            "annual_savings": base,
            "range_low": round(base * 0.7),
            "range_high": round(base * 1.3),
            "confidence": "Low — depends on Vendor B's willingness and ability to retrain; may require contract renegotiation",
            "assumptions": "Reduce Vendor B contacts/ticket to Vendor A level through training/QA intervention.",
        }


# ─────────────────────────────────────────────────────────────────
# Tool 5: get_weekly_trends
# ─────────────────────────────────────────────────────────────────
def get_weekly_trends():
    """Return week-over-week comparison for all key metrics."""
    df = _load_data()

    weeks = sorted(df["week"].unique())
    if len(weeks) < 2:
        return {"error": "Need at least 2 weeks for comparison"}

    latest = weeks[-1]
    prior = weeks[-2]
    df_latest = df[df["week"] == latest]
    df_prior = df[df["week"] == prior]

    def calc_metrics(subset):
        chatbot_subset = subset[subset["assigned_team"] == "ai_chatbot"]
        phone_count = (subset["channel"] == "phone").sum()
        # Containment = % of chatbot tickets resolved without escalation
        chatbot_containment = (
            round(chatbot_subset["is_resolved"].mean() * 100, 1)
            if len(chatbot_subset) > 0 else 0.0
        )
        return {
            "ticket_volume": len(subset),
            "total_cost": round(subset["cost_usd"].sum(), 2),
            "avg_cost_per_ticket": round(subset["cost_usd"].mean(), 2),
            "avg_csat": round(subset["csat_score"].mean(), 2),
            "resolution_rate": round(subset["is_resolved"].mean() * 100, 1),
            "escalation_rate": round(subset["is_escalated"].mean() * 100, 1),
            "abandonment_rate": round(subset["is_abandoned"].mean() * 100, 1),
            "chatbot_containment_pct": chatbot_containment,
            "phone_ticket_pct": round(phone_count / len(subset) * 100, 1),
            "avg_contacts": round(subset["contacts_per_ticket"].mean(), 2),
            "avg_first_response_min": round(subset["first_response_min"].mean(), 1),
        }

    m_latest = calc_metrics(df_latest)
    m_prior = calc_metrics(df_prior)

    # Build comparison with deltas
    comparison = {}
    for key in m_latest:
        curr = m_latest[key]
        prev = m_prior[key]
        delta = round(curr - prev, 2)
        pct_change = round((delta / abs(prev)) * 100, 1) if prev != 0 else 0.0

        # Determine direction
        improving_if_down = ["avg_cost_per_ticket", "escalation_rate", "abandonment_rate",
                             "avg_contacts", "avg_first_response_min", "phone_ticket_pct"]
        improving_if_up = ["avg_csat", "resolution_rate", "chatbot_containment_pct"]

        if abs(pct_change) < 1.0:
            direction = "flat"
        elif key in improving_if_down:
            direction = "improving" if delta < 0 else "worsening"
        elif key in improving_if_up:
            direction = "improving" if delta > 0 else "worsening"
        else:
            direction = "increasing" if delta > 0 else "decreasing"

        flagged = abs(pct_change) > 5.0 and direction == "worsening"

        comparison[key] = {
            "prior_week": prev,
            "current_week": curr,
            "delta": delta,
            "pct_change": pct_change,
            "direction": direction,
            "flagged": flagged,
        }

    return {
        "prior_week_num": int(prior),
        "current_week_num": int(latest),
        "comparison": comparison,
        "flagged_metrics": [k for k, v in comparison.items() if v["flagged"]],
    }


# ─────────────────────────────────────────────────────────────────
# Tool 6: analyze_customer_messages
# ─────────────────────────────────────────────────────────────────
def analyze_customer_messages():
    """NLP analysis on customer_message field: themes, emerging patterns, category keywords."""
    df = _load_data()

    # Clean text
    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.strip()

    df["clean_msg"] = df["customer_message"].apply(clean_text)

    # Top themes using TF-IDF bigrams — surfaces distinguishing themes, not just frequent ones
    tfidf = TfidfVectorizer(ngram_range=(2, 2), stop_words="english", max_features=200)
    X_tfidf = tfidf.fit_transform(df["clean_msg"])
    # Rank by mean TF-IDF score across documents (higher = more distinctive)
    tfidf_means = dict(zip(tfidf.get_feature_names_out(), X_tfidf.mean(axis=0).A1))
    # Also get raw counts for reporting
    count_vec = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=200)
    X_count = count_vec.fit_transform(df["clean_msg"])
    bigram_counts = dict(zip(count_vec.get_feature_names_out(), X_count.sum(axis=0).A1))
    # Sort by TF-IDF score but report counts too
    top_themes = sorted(tfidf_means.items(), key=lambda x: x[1], reverse=True)[:10]
    top_themes_list = [
        {
            "theme": theme,
            "tfidf_score": round(score, 4),
            "tickets": int(bigram_counts.get(theme, 0)),
            "pct_of_total": round(bigram_counts.get(theme, 0) / len(df) * 100, 1),
        }
        for theme, score in top_themes
    ]

    # Week-over-week comparison (W9 vs W10)
    weeks = sorted(df["week"].unique())
    w_prior = weeks[-2]
    w_latest = weeks[-1]
    df_prior = df[df["week"] == w_prior]
    df_latest = df[df["week"] == w_latest]

    # Latest-week-only themes (for dashboard display)
    tfidf_latest = TfidfVectorizer(ngram_range=(2, 2), stop_words="english", max_features=200)
    X_tfidf_latest = tfidf_latest.fit_transform(df_latest["clean_msg"])
    tfidf_latest_means = dict(zip(tfidf_latest.get_feature_names_out(), X_tfidf_latest.mean(axis=0).A1))
    count_latest = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=200)
    X_count_latest = count_latest.fit_transform(df_latest["clean_msg"])
    latest_counts = dict(zip(count_latest.get_feature_names_out(), X_count_latest.sum(axis=0).A1))
    top_latest = sorted(tfidf_latest_means.items(), key=lambda x: x[1], reverse=True)[:10]
    latest_week_themes = [
        {
            "theme": theme,
            "tfidf_score": round(score, 4),
            "tickets": int(latest_counts.get(theme, 0)),
            "pct_of_total": round(latest_counts.get(theme, 0) / len(df_latest) * 100, 1),
        }
        for theme, score in top_latest
    ]

    # Get bigram frequencies per week
    def get_bigrams(subset):
        vec = CountVectorizer(ngram_range=(2, 2), stop_words="english", max_features=200)
        X = vec.fit_transform(subset["clean_msg"])
        return dict(zip(vec.get_feature_names_out(), X.sum(axis=0).A1))

    prior_bigrams = get_bigrams(df_prior)
    latest_bigrams = get_bigrams(df_latest)

    # Find emerging patterns (grew >20% WoW), then deduplicate overlapping bigrams
    raw_emerging = []
    for bigram, latest_count in latest_bigrams.items():
        prior_count = prior_bigrams.get(bigram, 0)
        if prior_count >= 5:  # only compare if meaningful baseline
            growth = (latest_count - prior_count) / prior_count * 100
            if growth > 20:
                raw_emerging.append({
                    "theme": bigram,
                    "w_prior_count": int(prior_count),
                    "w_latest_count": int(latest_count),
                    "growth_pct": round(growth, 1),
                })
    raw_emerging = sorted(raw_emerging, key=lambda x: x["growth_pct"], reverse=True)

    # Deduplicate in two passes:
    # Pass 1: if two bigrams share a word, keep the one with higher growth
    deduped_pass1 = []
    used_words = set()
    for item in raw_emerging:
        words = set(item["theme"].split())
        if words & used_words:
            continue
        deduped_pass1.append(item)
        used_words.update(words)

    # Pass 2: if two themes have identical WoW counts, they're from the same
    # set of messages — keep only the first (highest growth already sorted)
    emerging = []
    seen_counts = set()
    for item in deduped_pass1:
        count_key = (item["w_prior_count"], item["w_latest_count"])
        if count_key in seen_counts:
            continue
        emerging.append(item)
        seen_counts.add(count_key)
        if len(emerging) >= 7:
            break

    # Top keywords per category
    category_keywords = {}
    for cat in df["category"].unique():
        cat_text = df[df["category"] == cat]["clean_msg"]
        vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50)
        X = vec.fit_transform(cat_text)
        counts = dict(zip(vec.get_feature_names_out(), X.sum(axis=0).A1))
        top3 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        category_keywords[cat] = [{"keyword": kw, "count": int(c)} for kw, c in top3]

    return {
        "top_themes": top_themes_list,
        "latest_week_themes": latest_week_themes,
        "emerging_patterns": emerging,
        "category_keywords": category_keywords,
        "prior_week": int(w_prior),
        "latest_week": int(w_latest),
    }


# ─────────────────────────────────────────────────────────────────
# Tool 7: generate_brief
# ─────────────────────────────────────────────────────────────────
def generate_brief(findings=None):
    """Generate the weekly brief. Uses findings from Claude if valid, otherwise calls tools directly."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Try to use Claude's findings; fall back to calling tools if malformed
    def _safe_get(key, fallback_fn):
        if findings and isinstance(findings, dict):
            val = findings.get(key)
            if isinstance(val, dict) and len(val) > 0:
                return val
        return fallback_fn()

    data_quality = _safe_get("data_quality", check_data_quality)
    anomaly_result = _safe_get("anomalies", flag_anomalies)
    nlp = _safe_get("nlp", analyze_customer_messages)

    # Weekly trends need stricter validation: the comparison dict must
    # contain all expected metric keys with nested prior_week/current_week
    # values.  Claude sometimes restructures or drops keys, so always
    # fall back to the deterministic tool when the data looks incomplete.
    _required_wow_keys = {
        "ticket_volume", "avg_cost_per_ticket", "avg_csat",
        "resolution_rate", "escalation_rate",
        "chatbot_containment_pct", "phone_ticket_pct",
    }
    trends_candidate = _safe_get("weekly_trends", get_weekly_trends)
    _comp = trends_candidate.get("comparison", {})
    if not _required_wow_keys.issubset(_comp.keys()) or not all(
        isinstance(_comp.get(k), dict)
        and "prior_week" in _comp[k]
        and "current_week" in _comp[k]
        for k in _required_wow_keys
    ):
        trends = get_weekly_trends()
    else:
        trends = trends_candidate

    # Opportunities: expect a list, fall back to calling all 5
    opp_names = ["chatbot_deflection", "agent_copilot", "urgent_routing", "phone_deflection", "bpo_vendor_b"]
    if findings and isinstance(findings, dict) and isinstance(findings.get("opportunities"), list):
        opportunities = findings["opportunities"]
        # Validate: each item must be a dict with "annual_savings"
        if not all(isinstance(o, dict) and "annual_savings" in o for o in opportunities):
            opportunities = [size_opportunity(name) for name in opp_names]
    else:
        opportunities = [size_opportunity(name) for name in opp_names]

    week_num = trends.get("current_week_num", "N/A")
    comparison = trends.get("comparison", {})
    anomalies = anomaly_result.get("anomalies", [])

    # Sort opportunities by annual_savings descending
    opportunities = sorted(opportunities, key=lambda x: x.get("annual_savings", 0), reverse=True)
    total_value = sum(o.get("annual_savings", 0) for o in opportunities)

    # Build brief
    lines = []
    lines.append(f"# Groupon Ops Intelligence Brief")
    lines.append(f"## Week {week_num} | Generated {now}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    top_anomaly = anomalies[0] if anomalies else None
    if top_anomaly:
        lines.append(f"This week's most critical finding: **{top_anomaly['title']}** — {top_anomaly['detail']} "
                      f"Across all identified opportunities, the total addressable value is **${total_value:,}/yr**.")
    lines.append("")

    # Top 5 Issues
    lines.append("## Top 5 Issues This Week")
    lines.append("*Ranked by annual business impact*")
    lines.append("")
    for i, opp in enumerate(opportunities[:5], 1):
        opp_name = opp["opportunity"]
        opp_id = OPPORTUNITY_ID_MAP.get(opp_name, "")
        meta = OPPORTUNITY_META.get(opp_id, {})
        owner = meta.get("owner", "Operations Leadership")
        action = meta.get("action", "Investigate and address root cause")
        timeline = meta.get("timeline", "TBD")
        kpi = meta.get("kpi", "")
        root_cause = meta.get("root_cause", opp.get("assumptions", "See detail"))
        range_low = opp.get("range_low", round(opp["annual_savings"] * 0.8))
        range_high = opp.get("range_high", round(opp["annual_savings"] * 1.2))
        confidence = opp.get("confidence", "")

        lines.append(f"{i}. **{opp_name}** — ${opp['annual_savings']:,}/yr (range: ${range_low:,}–${range_high:,})")
        lines.append(f"   Root cause: {root_cause}")
        lines.append(f"   Action: {action}")
        lines.append(f"   Owner: {owner} | Timeline: {timeline}")
        if kpi:
            lines.append(f"   KPI target: {kpi}")
        if confidence:
            lines.append(f"   Confidence: {confidence}")
        lines.append("")

    # Week-over-Week Comparison
    lines.append("## Week-over-Week Comparison")
    lines.append("")

    # Build WoW table
    prior_w = trends.get("prior_week_num", "N-1")
    curr_w = trends.get("current_week_num", "N")
    lines.append(f"| Metric | W{prior_w} | W{curr_w} | Delta | Trend |")
    lines.append("|---|---|---|---|---|")

    metric_labels = {
        "ticket_volume": "Ticket volume",
        "avg_cost_per_ticket": "Avg cost/ticket",
        "avg_csat": "CSAT score",
        "resolution_rate": "Resolution rate",
        "escalation_rate": "Escalation rate",
        "chatbot_containment_pct": "Chatbot containment",
        "phone_ticket_pct": "Phone ticket share",
    }

    for key, label in metric_labels.items():
        data = comparison.get(key, {})
        prev = data.get("prior_week", "—")
        curr = data.get("current_week", "—")
        pct = data.get("pct_change", 0)
        direction = data.get("direction", "—")

        # Format values
        if "cost" in key:
            prev_str, curr_str = f"${prev}", f"${curr}"
        elif "rate" in key or "pct" in key:
            prev_str, curr_str = f"{prev}%", f"{curr}%"
        else:
            prev_str, curr_str = f"{prev:,}" if isinstance(prev, int) else str(prev), \
                                 f"{curr:,}" if isinstance(curr, int) else str(curr)

        delta_str = f"{pct:+.1f}%"
        trend_str = direction.capitalize()
        lines.append(f"| {label} | {prev_str} | {curr_str} | {delta_str} | {trend_str} |")

    lines.append("")

    # Recommended Actions — pull from OPPORTUNITY_META, grouped by timeline
    lines.append("## Recommended Actions This Week")
    lines.append("")
    timeline_order = ["IMMEDIATE (this week)", "SHORT TERM (this month)", "MEDIUM TERM (this quarter)"]
    for timeline in timeline_order:
        actions_in_tier = []
        for opp in opportunities[:5]:
            opp_id = OPPORTUNITY_ID_MAP.get(opp["opportunity"], "")
            meta = OPPORTUNITY_META.get(opp_id, {})
            if meta.get("timeline") == timeline:
                actions_in_tier.append(meta)
        for meta in actions_in_tier:
            lines.append(f"- **{timeline}**: {meta['action']} — Owner: {meta['owner']}")
    lines.append("")

    # Watch List
    lines.append("## Watch List — Emerging Patterns")
    lines.append("")

    # Context-aware "why it matters" for common customer themes
    THEME_CONTEXT = {
        "double charged": "Billing errors erode trust fast — high churn risk if not resolved quickly",
        "charged purchase": "Unexpected charges are a top driver of chargebacks and negative reviews",
        "refund waiting": "Refund delays are the #1 CSAT detractor across all channels",
        "voucher expired": "Expired voucher complaints spike before promo deadlines — may need policy review",
        "email address": "Account access issues block customers from using purchases — drives repeat contacts",
        "change email": "Spike in email change requests may indicate account security concerns",
        "address account": "Account update friction increases abandonment and repeat contacts",
        "merchant closed": "Merchant closures create unrecoverable customer experiences — needs proactive detection",
        "wrong item": "Fulfillment errors require costly returns and re-ships",
        "credit card": "Payment method issues directly block revenue",
        "cancelled subscription": "Subscription billing complaints signal retention risk",
        "money back": "Refund demand language indicates high customer frustration",
    }
    default_context = "If this trend continues, it could become a top-5 issue within 2-3 weeks"

    # From NLP emerging patterns
    emerging = nlp.get("emerging_patterns", [])
    for pattern in emerging[:5]:
        theme = pattern["theme"]
        context = THEME_CONTEXT.get(theme, default_context)
        lines.append(f"- **{theme}**: grew {pattern['growth_pct']:.0f}% WoW "
                      f"(W{nlp.get('prior_week', trends.get('prior_week_num', '?'))}: {pattern['w_prior_count']} → "
                      f"W{nlp.get('latest_week', trends.get('current_week_num', '?'))}: {pattern['w_latest_count']} mentions)")
        lines.append(f"  Why it matters: {context}")
        lines.append("")

    # From flagged metrics
    flagged = trends.get("flagged_metrics", [])
    for metric_key in flagged:
        data = comparison.get(metric_key, {})
        label = metric_labels.get(metric_key, metric_key)
        lines.append(f"- **{label}**: moved {data.get('pct_change', 0):+.1f}% "
                      f"({data.get('prior_week', '?')} → {data.get('current_week', '?')})")
        lines.append(f"  Why it matters: Exceeds 5% threshold for watch list flagging")
        lines.append("")

    # Data Quality Notes
    lines.append("## Data Quality Notes")
    lines.append("")
    if data_quality:
        row_count = data_quality.get("row_count", 0)
        missing_csat = data_quality.get("missing_csat_rate", 0)
        lines.append(f"- Total clean tickets analyzed: {row_count:,}" if isinstance(row_count, int) else f"- Total clean tickets analyzed: {row_count}")
        lines.append(f"- Missing CSAT rate: {missing_csat}%")
        for note in data_quality.get("anomalies_found", []):
            lines.append(f"- {note}")
    lines.append("")

    brief_text = "\n".join(lines)

    # Save to file
    output_path = ROOT / "output" / "weekly_brief.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(brief_text)

    return {"brief": brief_text, "saved_to": str(output_path)}


# ─────────────────────────────────────────────────────────────────
# Tool schemas for Anthropic API
# ─────────────────────────────────────────────────────────────────
TOOL_SCHEMAS = [
    {
        "name": "check_data_quality",
        "description": "Load the ticket CSV and return a data quality summary: row count, missing values, anomalies found.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "analyze_metric",
        "description": "Group ticket data by a dimension and compute summary statistics for a metric. Use this to compare teams, channels, categories, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "description": "The metric to analyze. Valid: cost_usd, csat_score, resolution_min, first_response_min, contacts_per_ticket, is_resolved, is_escalated, is_abandoned",
                },
                "group_by": {
                    "type": "string",
                    "description": "The dimension to group by. Valid: assigned_team, channel, category, priority, market, week",
                },
            },
            "required": ["metric", "group_by"],
        },
    },
    {
        "name": "flag_anomalies",
        "description": "Run anomaly detection across all key metrics. Checks routing errors, escalation rates, CSAT outliers, cost anomalies, and high-touch concentration.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "size_opportunity",
        "description": "Calculate the annual savings for a named improvement opportunity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "opportunity_name": {
                    "type": "string",
                    "description": "The opportunity to size. Valid: chatbot_deflection, agent_copilot, urgent_routing, phone_deflection, bpo_vendor_b",
                },
            },
            "required": ["opportunity_name"],
        },
    },
    {
        "name": "get_weekly_trends",
        "description": "Return week-over-week comparison for all key metrics between the two most recent weeks. Flags any metric that moved more than 5% in the wrong direction.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "analyze_customer_messages",
        "description": "Run NLP analysis on the customer_message field. Extracts top themes (bigrams), compares keyword frequency between most recent week and prior week to find emerging patterns growing >20% WoW, and returns top keywords per category.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "generate_brief",
        "description": "Generate the final weekly Ops Intelligence Brief in markdown. Pass ALL findings from prior tool calls as input. If the data is malformed, the tool will re-run analyses automatically. The brief includes: executive summary, top 5 issues, WoW comparison table, recommended actions with owners, watch list of emerging patterns, and data quality notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "object",
                    "description": "Structured dict with keys: data_quality, anomalies, opportunities (list), weekly_trends, nlp. Each should contain the raw output from the corresponding tool call.",
                },
            },
            "required": ["findings"],
        },
    },
]


# Map tool names to functions
TOOL_FUNCTIONS = {
    "check_data_quality": lambda args: check_data_quality(),
    "analyze_metric": lambda args: analyze_metric(args["metric"], args["group_by"]),
    "flag_anomalies": lambda args: flag_anomalies(),
    "size_opportunity": lambda args: size_opportunity(args["opportunity_name"]),
    "get_weekly_trends": lambda args: get_weekly_trends(),
    "analyze_customer_messages": lambda args: analyze_customer_messages(),
    "generate_brief": lambda args: generate_brief(args.get("findings")),
}
