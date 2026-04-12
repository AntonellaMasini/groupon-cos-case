"""
Unit tests for agent tools.
Run with: python -m pytest tests/test_tools.py -v
"""

import sys
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.tools import (
    check_data_quality,
    size_opportunity,
    generate_brief,
    get_weekly_trends,
    analyze_customer_messages,
    flag_anomalies,
    invalidate_cache,
)


class TestCheckDataQuality:
    """Tests for the check_data_quality tool."""

    def test_returns_expected_keys(self):
        result = check_data_quality()
        expected_keys = {
            "row_count", "columns", "weeks", "missing_values",
            "missing_pct", "anomalies_found", "urgent_chatbot_count",
            "missing_csat_rate", "high_touch_count",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_row_count_positive(self):
        result = check_data_quality()
        assert result["row_count"] > 0

    def test_weeks_are_sorted(self):
        result = check_data_quality()
        assert result["weeks"] == sorted(result["weeks"])

    def test_anomalies_found_is_list(self):
        result = check_data_quality()
        assert isinstance(result["anomalies_found"], list)


class TestSizeOpportunity:
    """Tests for the size_opportunity tool."""

    def test_rejects_invalid_name(self):
        result = size_opportunity("nonexistent_opportunity")
        assert "error" in result

    def test_chatbot_deflection_has_required_fields(self):
        result = size_opportunity("chatbot_deflection")
        assert "error" not in result
        for field in ["opportunity", "annual_savings", "range_low", "range_high", "confidence"]:
            assert field in result, f"Missing field: {field}"

    def test_annual_savings_positive(self):
        valid_names = [
            "chatbot_deflection", "agent_copilot", "urgent_routing",
            "phone_deflection", "bpo_vendor_b",
        ]
        for name in valid_names:
            result = size_opportunity(name)
            assert result["annual_savings"] > 0, f"{name} has non-positive savings"

    def test_confidence_range_bounds(self):
        result = size_opportunity("chatbot_deflection")
        assert result["range_low"] < result["annual_savings"]
        assert result["range_high"] > result["annual_savings"]

    def test_all_opportunities_have_confidence(self):
        valid_names = [
            "chatbot_deflection", "agent_copilot", "urgent_routing",
            "phone_deflection", "bpo_vendor_b",
        ]
        for name in valid_names:
            result = size_opportunity(name)
            assert "confidence" in result and len(result["confidence"]) > 0, (
                f"{name} missing confidence field"
            )


class TestGenerateBrief:
    """Tests for the generate_brief tool."""

    def test_brief_contains_all_four_sections(self):
        result = generate_brief()
        brief_text = result["brief"]
        assert "Top 5 Issues" in brief_text, "Missing 'Top 5 Issues' section"
        assert "Week-over-Week" in brief_text, "Missing 'Week-over-Week' section"
        assert "Recommended Actions" in brief_text, "Missing 'Recommended Actions' section"
        assert "Watch List" in brief_text, "Missing 'Watch List' section"

    def test_brief_has_executive_summary(self):
        result = generate_brief()
        assert "Executive Summary" in result["brief"]

    def test_brief_includes_root_cause(self):
        result = generate_brief()
        assert "Root cause:" in result["brief"]

    def test_brief_includes_confidence_ranges(self):
        result = generate_brief()
        # The brief should contain range notation like "$X-$Y"
        assert "range:" in result["brief"].lower() or "\u2013" in result["brief"]

    def test_brief_lists_all_five_opportunities(self):
        result = generate_brief()
        brief = result["brief"]
        expected_opps = [
            "chatbot deflection", "co-pilot", "urgent",
            "phone", "Vendor B",
        ]
        for keyword in expected_opps:
            assert keyword.lower() in brief.lower(), (
                f"Brief missing opportunity containing '{keyword}'"
            )


class TestGetWeeklyTrends:
    """Tests for the get_weekly_trends tool."""

    def test_returns_comparison_data(self):
        result = get_weekly_trends()
        assert "comparison" in result
        assert "current_week_num" in result
        assert "prior_week_num" in result

    def test_comparison_has_key_metrics(self):
        result = get_weekly_trends()
        comparison = result["comparison"]
        expected_metrics = [
            "ticket_volume", "avg_cost_per_ticket", "avg_csat",
            "resolution_rate", "escalation_rate",
        ]
        for metric in expected_metrics:
            assert metric in comparison, f"Missing metric: {metric}"


class TestFlagAnomalies:
    """Tests for the flag_anomalies tool."""

    def test_returns_anomalies_list(self):
        result = flag_anomalies()
        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)

    def test_anomalies_have_required_fields(self):
        result = flag_anomalies()
        for anomaly in result["anomalies"]:
            for field in ["id", "severity", "title", "detail"]:
                assert field in anomaly, f"Anomaly missing field: {field}"

    def test_urgent_routing_detected(self):
        """The dataset has >50 urgent tickets routed to chatbot; this should be flagged."""
        result = flag_anomalies()
        ids = [a["id"] for a in result["anomalies"]]
        assert "urgent_routing" in ids


class TestCaching:
    """Tests for the data caching mechanism."""

    def test_invalidate_cache_works(self):
        # First call loads data
        result1 = check_data_quality()
        # Invalidate and reload
        invalidate_cache()
        result2 = check_data_quality()
        assert result1["row_count"] == result2["row_count"]
