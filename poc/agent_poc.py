#!/usr/bin/env python3
"""
Minimal PK agent POC (stdlib only).
Orchestrates a multi-agent loop with stop criteria.
"""

from __future__ import annotations

from pathlib import Path

from .agents import (
    AgentState,
    agent_fit_individual,
    agent_fit_pooled,
    agent_inspect,
    agent_read_paper,
    agent_report,
    should_stop,
)
from .io_utils import read_rows
from .model import group_by_subject


DATA_CSV = Path("data/pkpd_arnaud_Acocella_1984_Data - Acocella 1984 Data.csv")
META_JSON = Path("data/Acocella 1984 Metadata.json")
OUT_RESULTS = Path("poc/results.csv")
OUT_REPORT = Path("poc/report.md")
PAPER_PDF = Path("data/Public Data for Local Test.pdf")

MAX_ITERATIONS = 3
NO_IMPROVEMENT_STOP = 2
ANTHROPIC_MODEL = "claude-3-5-sonnet-latest"
LOCAL_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def run_agent_loop(state: AgentState) -> None:
    """Run the agent loop with stop criteria."""
    agents = [agent_inspect, agent_fit_individual, agent_fit_pooled]
    iteration = 0
    while True:
        for agent in agents:
            agent(state)
        provider = _llm_provider()
        model_name = ANTHROPIC_MODEL if provider == "anthropic" else _local_model()
        agent_read_paper(state, PAPER_PDF, _anthropic_key(), model_name, provider)
        agent_report(state, OUT_RESULTS, OUT_REPORT, META_JSON)
        if should_stop(state, iteration, MAX_ITERATIONS, NO_IMPROVEMENT_STOP):
            break
        iteration += 1


def main() -> None:
    """Entry point: load data, build state, run agent loop."""
    rows = read_rows(DATA_CSV)
    by_subject = group_by_subject(rows)
    state = AgentState(rows=rows, by_subject=by_subject)
    run_agent_loop(state)


def _anthropic_key() -> str:
    """Read Anthropic API key from environment."""
    import os

    return os.environ.get("ANTHROPIC_API_KEY", "")


def _llm_provider() -> str:
    """Return the LLM provider: 'anthropic' or 'local'."""
    import os

    return os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()


def _local_model() -> str:
    """Read local model name from env or fallback default."""
    import os

    return os.environ.get("LOCAL_MODEL_NAME", LOCAL_MODEL)


if __name__ == "__main__":
    main()
