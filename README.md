# PKPD Agent POC (Acocella 1984)

This repository contains a minimal PK agent POC that, from raw data and a few task hints, rebuilds a simple PK model and generates a short report.

POC goals (very simple):
- Basic data inspection
- One‑compartment IV infusion PK model formulation
- Rough fitting (grid search)
- Simple comparison and report/CSV export

## Quick start

```bash
uv venv
uv pip install -e .
export ANTHROPIC_API_KEY="your_key_here"
python3 -m poc.agent_poc
```

Outputs:
- `poc/results.csv` (per‑subject estimated parameters)
- `poc/report.md` (short POC report, includes paper insights if LLM enabled)

## Parallel agents

You can run the paper-reading agent in parallel with the fitting steps:

```bash
export PARALLEL_AGENTS=1
python3 -m poc.agent_poc
```

## Validation (simple)

Generate residuals, a short summary, and optional plots:

```bash
python3 -m poc.validate
```

Outputs:
- `poc/validation_residuals.csv`
- `poc/validation_summary.md`
- `poc/plots/obs_vs_pred.png` (if matplotlib is installed)

## Local LLM option (HF)

You can switch to a local Hugging Face model for paper extraction:

```bash
export LLM_PROVIDER="local"
export LOCAL_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
uv pip install -e ".[local]"
python3 -m poc.agent_poc
```

Notes:
- Requires `transformers` and `bitsandbytes` installed.
- Local models are usually less reliable for structured extraction.

## Structure

- `data/`: data + metadata + reference PDF
- `poc/agent_poc.py`: standalone POC agent (stdlib only)
- `poc/steps.md`: step‑by‑step agent guidelines

## POC limitations

- No population estimation (no mixed effects)
- Grid‑search fit (slow but dependency‑free)
- No graphical diagnostics (no plots)
