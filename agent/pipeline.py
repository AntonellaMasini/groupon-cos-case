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

# Add project root and agent dir to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))

from agent.tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

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
customer support ticket data and produce a weekly Ops Intelligence Brief for the Chief of Staff.

You have access to 7 tools. Use your judgment about which analyses to run and in what order
based on what the data tells you. Start by understanding data quality, then explore the data
to find the most significant operational issues.

Your brief MUST contain all four of these sections:
1. Top 5 issues this week (ranked by annual business impact)
2. Week-over-week comparison (explicit numbers: prior week vs current week)
3. Recommended actions with named owners and timelines
4. Watch list of emerging patterns (things trending badly but not yet top 5)

Key behaviors that make you an effective agent:
- After running anomaly detection, REVIEW the results. If something unexpected surfaces,
  use analyze_metric to drill deeper before moving on. Investigate what the data tells you.
- If NLP themes correlate with a spiking metric, call that out in the brief — connect the dots.
- Size every opportunity you discover. The brief should quantify annual savings with ranges.
- When compiling the final brief via generate_brief, pass ALL prior findings as a structured
  dict with keys: data_quality, anomalies, weekly_trends, nlp, opportunities (list).

Available opportunities to size: chatbot_deflection, agent_copilot, urgent_routing,
phone_deflection, bpo_vendor_b. Size all of them.

Be thorough but efficient. Think before each tool call — explain what you expect to find
and why you're running that analysis."""


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
                "Start by understanding the data, then investigate the most significant issues. "
                "Size all opportunities, and compile everything into the final brief. "
                "Use your judgment about what to explore deeper based on what you find."
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
                log_step("thinking", block.text[:500])

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
