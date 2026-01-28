# Agent guidelines (POC)

Goal: reproduce a minimal PK workflow from raw data, with no external dependencies.

## 1) Data inspection
- Load the raw CSV.
- Check expected columns (ID, TIME, CONC, Dose, Condition).
- Summarize: n subjects, n observations, min/max time, doses and infusion durations.

## 2) Model formulation
- One‑compartment IV infusion PK model, first‑order elimination.
- Parameters: CL (clearance), V (volume), k = CL/V.
- Infusion duration inferred from `Condition`.

## 3) Implementation & fit
- For each subject, minimize SSE between Cobs and Cpred.
- Use grid search (CL, V) to stay simple (stdlib only).
- Export estimated parameters + RMSE.

## 4) Quick comparison
- Fit a "pooled" model (all subjects) for a rough comparison.
- Report pooled SSE/RMSE vs individual.

## 5) Report
- Write a short `report.md` (data, model, results, limitations).

## 6) Possible extensions
- Nonlinear fit (SciPy), CIs, diagnostics
- More complex models (two‑compartment)
- RAG + auto‑extraction of the paper for comparison
