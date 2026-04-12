"""
Agentic pipeline for Groupon Ops Intelligence.
Uses the raw Anthropic Python SDK with tool use — no LangChain.
The agent decides which tools to call, in what order, based on the data.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

load_dotenv(ROOT / ".env")

# Pipeline log — records each step for Streamlit Tab 4
pipeline_log = []


def log_step(step_type, detail):
    """Log an agent step for the pipeline trace."""
    pipeline_log.append({"type": step_type, "detail": detail})
    prefix = {"tool_call": "TOOL", "tool_result": "RESULT", "thinking": "AGENT", "final": "BRIEF"}
    tag = prefix.get(step_type, "INFO")
    print(f"  [{tag}] {detail[:120]}")


SYSTEM_PROMPT = """You are the Groupon Operations Intelligence Agent. Your job is to analyze
customer support ticket data and produce a weekly Ops Intelligence Brief.

You have access to 7 tools. Use them to build a complete picture of operations this week.

Follow this analysis sequence:
1. First, check data quality to understand what you're working with.
2. Flag anomalies across all metrics.
3. Get weekly trends to see what changed week over week.
4. Run NLP analysis on customer messages to find emerging themes.
5. Size each opportunity: chatbot_deflection, agent_copilot, urgent_routing, phone_deflection, bpo_vendor_b.
6. Finally, compile ALL findings and call generate_brief with the complete findings dict.

When calling generate_brief, pass a findings dict with these keys:
- data_quality: output from check_data_quality
- anomalies: output from flag_anomalies
- weekly_trends: output from get_weekly_trends
- nlp: output from analyze_customer_messages
- opportunities: list of outputs from all size_opportunity calls

Be thorough. Call every tool. The brief must contain all four required sections:
top 5 issues, WoW comparison, recommended actions with owners, and watch list.

Do NOT skip any tools — each one contributes to a section of the brief.

IMPORTANT: After running flag_anomalies, review the results. The tool includes both
hardcoded checks AND statistical anomaly detection (z-score outliers, WoW spikes).
If the statistical detection surfaces something interesting that wasn't in the hardcoded
checks, use analyze_metric to dig deeper before generating the brief. This is what makes
you an agent — you investigate what the data tells you, not just follow a script."""


def run_agent():
    """Run the full agentic pipeline."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    print("── Groupon Ops Intelligence Agent ──────────────────────")
    print("  Starting analysis pipeline...\n")

    messages = [
        {
            "role": "user",
            "content": (
                "Analyze the latest ticket data and produce the weekly Ops Intelligence Brief. "
                "Follow your full analysis sequence: data quality → anomalies → trends → "
                "NLP themes → opportunity sizing → generate brief. "
                "Call every tool and compile all findings into the final brief."
            ),
        }
    ]

    # Agent loop — keep going until the agent produces a final text response
    max_iterations = 20
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        log_step("thinking", f"Iteration {iteration} — sending to Claude...")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Process response content blocks
        assistant_content = response.content
        tool_results = []

        for block in assistant_content:
            if block.type == "text":
                log_step("thinking", block.text[:200])

            elif block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_id = block.id

                log_step("tool_call", f"{tool_name}({json.dumps(tool_input)[:100]})")

                # Execute the tool
                if tool_name in TOOL_FUNCTIONS:
                    try:
                        result = TOOL_FUNCTIONS[tool_name](tool_input)
                        result_str = json.dumps(result, default=str)
                        log_step("tool_result", f"{tool_name} → {len(result_str)} chars")
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                        log_step("tool_result", f"{tool_name} → ERROR: {e}")
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})
                    log_step("tool_result", f"Unknown tool: {tool_name}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_str,
                })

        # Add assistant message to conversation
        messages.append({"role": "assistant", "content": assistant_content})

        # If there were tool calls, add results and continue the loop
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool calls — agent is done
            break

        # Check stop reason
        if response.stop_reason == "end_turn" and not tool_results:
            break

    # Extract final text
    final_text = ""
    for block in assistant_content:
        if block.type == "text":
            final_text += block.text

    print(f"\n── Pipeline Complete ────────────────────────────────────")
    print(f"  Iterations: {iteration}")
    print(f"  Tool calls: {sum(1 for l in pipeline_log if l['type'] == 'tool_call')}")

    # Check if brief was generated
    brief_path = ROOT / "output" / "weekly_brief.md"
    if brief_path.exists():
        print(f"  Brief saved: {brief_path}")
    else:
        print("  WARNING: Brief was not generated. Check agent output above.")

    return {
        "pipeline_log": pipeline_log,
        "final_text": final_text,
        "iterations": iteration,
    }


def get_pipeline_log():
    """Return the pipeline log for Streamlit Tab 4."""
    return pipeline_log


if __name__ == "__main__":
    run_agent()
