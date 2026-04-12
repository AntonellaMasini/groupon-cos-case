"""
Data cleaning script for Groupon support ticket data.
Loads raw CSV, applies cleaning steps, prints quality report, saves clean CSV.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "option_a_ticket_data.csv"
CLEAN_PATH = ROOT / "data" / "tickets_clean.csv"


def main():
    # ── 1. Inspect raw data ──────────────────────────────────────
    print("── STEP 1: Inspect raw data ─────────────────────────────")
    df = pd.read_csv(RAW_PATH)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")
    print("Dtypes:")
    print(df.dtypes.to_string())
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"  {col:<25} {count:>6}  ({pct:.1f}%)")

    # ── 2. Fix market label inconsistencies ──────────────────────
    print("\n── STEP 2: Fix market labels ────────────────────────────")
    market_map = {"United Kingdom": "UK", "GER": "DE", "USA": "US"}
    mask = df["market"].isin(market_map.keys())
    market_fixes = mask.sum()
    df["market"] = df["market"].replace(market_map)
    print(f"Market labels fixed: {market_fixes} rows")
    print(f"Unique markets after fix: {sorted(df['market'].dropna().unique())}")

    # ── 3. Fix CSAT scores (-1 → NaN) ───────────────────────────
    print("\n── STEP 3: Fix CSAT scores ──────────────────────────────")
    csat_neg = (df["csat_score"] == -1).sum()
    df.loc[df["csat_score"] == -1, "csat_score"] = np.nan
    print(f"CSAT -1 values replaced with NaN: {csat_neg}")

    # ── 4. Fix negative resolution times ─────────────────────────
    print("\n── STEP 4: Fix negative resolution times ────────────────")
    neg_res = (df["resolution_min"] < 0).sum()
    df.loc[df["resolution_min"] < 0, "resolution_min"] = np.nan
    print(f"Negative resolution_min set to NaN: {neg_res}")

    # ── 5. Parse dates and extract time features ─────────────────
    print("\n── STEP 5: Parse dates and extract time features ────────")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["week"] = df["created_at"].dt.isocalendar().week.astype(int)
    df["hour"] = df["created_at"].dt.hour
    df["day_of_week"] = df["created_at"].dt.day_name()
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")
    print(f"Weeks found: {sorted(df['week'].unique())}")

    # ── 6. Exclude partial week ──────────────────────────────────
    print("\n── STEP 6: Exclude partial week ─────────────────────────")
    week_counts_before = df["week"].value_counts().sort_index()
    print("Ticket count per week (before):")
    for wk, cnt in week_counts_before.items():
        print(f"  W{wk}: {cnt:,}")

    # Dynamically exclude partial weeks (< 50% of median week volume)
    week_counts = df["week"].value_counts()
    median_count = week_counts.median()
    valid_weeks = week_counts[week_counts > median_count * 0.5].index
    df = df[df["week"].isin(valid_weeks)].copy()

    week_counts_after = df["week"].value_counts().sort_index()
    print("\nTicket count per week (after, W7-W10 only):")
    for wk, cnt in week_counts_after.items():
        print(f"  W{wk}: {cnt:,}")
    print(f"Total clean tickets: {len(df):,}")

    # ── 7. Add helper columns ────────────────────────────────────
    print("\n── STEP 7: Add helper columns ──────────────────────────")
    df["is_resolved"] = df["resolution_status"] == "resolved"
    df["is_escalated"] = df["resolution_status"] == "escalated"
    df["is_abandoned"] = df["resolution_status"] == "abandoned"
    df["is_high_touch"] = df["contacts_per_ticket"] > 5
    print(f"is_resolved:   {df['is_resolved'].sum():,}")
    print(f"is_escalated:  {df['is_escalated'].sum():,}")
    print(f"is_abandoned:  {df['is_abandoned'].sum():,}")
    print(f"is_high_touch: {df['is_high_touch'].sum():,}")

    # ── 8. Data quality report ───────────────────────────────────
    raw_count = len(pd.read_csv(RAW_PATH))
    clean_count = len(df)
    missing_subcat = df["subcategory"].isnull().sum()
    missing_csat = df["csat_score"].isnull().sum()
    urgent_chatbot = ((df["priority"] == "urgent") & (df["assigned_team"] == "ai_chatbot")).sum()
    high_touch = df["is_high_touch"].sum()

    print("\n── DATA QUALITY REPORT ──────────────────────")
    print(f"Raw tickets:                {raw_count:,}")
    print(f"Clean tickets (W7-W10):     {clean_count:,}")
    print(f"Missing subcategory:        {missing_subcat:,} ({missing_subcat/clean_count*100:.1f}%)")
    print(f"Missing CSAT:               {missing_csat:,} ({missing_csat/clean_count*100:.1f}%)")
    print(f"CSAT -1 values fixed:       {csat_neg}")
    print(f"Negative resolution times:  {neg_res} set to NaN")
    print(f"Market labels fixed:        {market_fixes}")
    print(f"Urgent to chatbot anomaly:  {urgent_chatbot}")
    print(f"High-touch tickets (>5):    {high_touch:,} ({high_touch/clean_count*100:.1f}%)")
    # Data quality score: penalize for missing data and anomalies
    quality = 100
    quality -= int(missing_subcat / clean_count * 100 * 0.5)  # missing subcategory penalty
    quality -= int(missing_csat / clean_count * 100 * 0.5)    # missing CSAT penalty
    quality -= min(5, int(urgent_chatbot / 50))                # routing anomaly penalty
    quality -= min(3, neg_res)                                 # negative resolution penalty
    print(f"Data quality score:         {quality}/100")
    print("─────────────────────────────────────────────")

    # ── 9. Save cleaned data ─────────────────────────────────────
    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned data saved to {CLEAN_PATH}")
    print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()
