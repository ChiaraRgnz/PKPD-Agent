#!/usr/bin/env python3
"""
Simple validation utilities for the PK model.
Produces residual summaries and optional plots if matplotlib is installed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List

from .io_utils import read_rows
from .model import group_by_subject, predict_conc, grid_search, Row


DATA_CSV = Path("data/pkpd_acocella_1984_data.csv")
OUT_RESIDUALS = Path("poc/validation_residuals.csv")
OUT_SUMMARY = Path("poc/validation_summary.md")
OUT_PLOTS_DIR = Path("poc/plots")


def _fit_per_subject(rows: List[Row]) -> Dict[str, Dict[str, float]]:
    by_subject = group_by_subject(rows)
    out: Dict[str, Dict[str, float]] = {}
    for subject_id, srows in sorted(by_subject.items()):
        sse, cl, v = grid_search(srows)
        rmse = math.sqrt(sse / max(len(srows), 1))
        out[subject_id] = {"cl": cl, "v": v, "rmse": rmse}
    return out


def _write_residuals(rows: List[Row], params: Dict[str, Dict[str, float]]) -> None:
    OUT_RESIDUALS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RESIDUALS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["subject_id", "time_h", "conc_obs", "conc_pred", "residual"]
        )
        for r in rows:
            p = params[r.subject_id]
            pred = predict_conc(r.time, r.dose_mg, r.tinf_h, p["cl"], p["v"])
            writer.writerow([r.subject_id, r.time, r.conc, pred, r.conc - pred])


def _write_summary(rows: List[Row], params: Dict[str, Dict[str, float]]) -> None:
    rmses = [p["rmse"] for p in params.values()]
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        f.write("# Validation summary\n\n")
        f.write(f"- Subjects: {len(params)}\n")
        f.write(f"- Observations: {len(rows)}\n")
        f.write(f"- RMSE (min/median/max): {min(rmses):.3g} / ")
        f.write(f"{sorted(rmses)[len(rmses)//2]:.3g} / {max(rmses):.3g}\n")
        f.write(f"- Residuals file: {OUT_RESIDUALS}\n")


def _try_plots(rows: List[Row], params: Dict[str, Dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    OUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    # Observed vs predicted scatter
    obs = []
    pred = []
    for r in rows:
        p = params[r.subject_id]
        pred.append(predict_conc(r.time, r.dose_mg, r.tinf_h, p["cl"], p["v"]))
        obs.append(r.conc)
    plt.figure()
    plt.scatter(obs, pred, s=12, alpha=0.7)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Observed vs Predicted")
    plt.savefig(OUT_PLOTS_DIR / "obs_vs_pred.png", dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    rows = read_rows(DATA_CSV)
    params = _fit_per_subject(rows)
    _write_residuals(rows, params)
    _write_summary(rows, params)
    _try_plots(rows, params)


if __name__ == "__main__":
    main()
