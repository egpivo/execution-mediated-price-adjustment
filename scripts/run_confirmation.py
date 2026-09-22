#!/usr/bin/env python3
"""PHASE I: the one frozen confirmatory execution.

Primary model (CONFIRMATORY_SPEC.json, no change-point search):
    C_i = b0 + b1*x_i + b2*(x_i-1)_+ + eps_i
on focal-swap observations only (S_n=1), breakpoint fixed at x=1.

Inference: calendar-day cluster bootstrap, 2000 reps, seed 20260829,
percentile 2.5/97.5 interval -- same convention as the discovery sample's
run_service_bootstrap.py, reused because no technical defect in it was
found (CONFIRMATORY_SPEC.json's stated condition for changing it).

Precomputed per-day sufficient statistics (X'X, X'y per day) make the
bootstrap fast; a direct-WLS self-check on the point estimate verifies the
aggregation is equivalent before trusting any bootstrap replicate, mirroring
the discovery pipeline's own verification discipline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPS = 2000
SEED = 20260829


def design(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x, np.clip(x - 1.0, 0.0, None)])


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def per_day_suffstats(day_ids: np.ndarray, X: np.ndarray, y: np.ndarray):
    days = np.unique(day_ids)
    XtX = {}
    Xty = {}
    for d in days:
        m = day_ids == d
        Xd = X[m]
        XtX[d] = Xd.T @ Xd
        Xty[d] = Xd.T @ y[m]
    return days, XtX, Xty


def bootstrap(days: np.ndarray, XtX: dict, Xty: dict, X: np.ndarray, y: np.ndarray,
              day_ids: np.ndarray, reps: int, seed: int) -> np.ndarray:
    # self-check: aggregated suffstats reproduce direct OLS point estimate
    XtX_sum = sum(XtX.values())
    Xty_sum = sum(Xty.values())
    beta_agg = np.linalg.solve(XtX_sum, Xty_sum)
    beta_direct = fit_ols(X, y)
    assert np.allclose(beta_agg, beta_direct, atol=1e-6), "suffstat aggregation mismatch"

    rng = np.random.default_rng(seed)
    n_days = days.size
    betas = np.zeros((reps, 3))
    for r in range(reps):
        draw = rng.choice(days, size=n_days, replace=True)
        counts = pd.Series(draw).value_counts()
        XtX_b = np.zeros((3, 3))
        Xty_b = np.zeros(3)
        for d, c in counts.items():
            XtX_b += c * XtX[d]
            Xty_b += c * Xty[d]
        betas[r] = np.linalg.solve(XtX_b, Xty_b)
    return betas


def run_market(panel_path: Path, seconds_per_day: int, label: str) -> dict:
    t = pq.read_table(panel_path, columns=["block_timestamp", "retained_hf", "has_focal_swap", "x", "correction_hf"]).to_pandas()
    sub = t[t.retained_hf & t.has_focal_swap].copy()
    sub = sub[np.isfinite(sub.x) & np.isfinite(sub.correction_hf)]
    sub["day"] = (sub.block_timestamp.astype("int64") // 86400).astype("int64")

    X = design(sub.x.to_numpy())
    y = sub.correction_hf.to_numpy()
    day_ids = sub.day.to_numpy()

    beta_point = fit_ols(X, y)
    days, XtX, Xty = per_day_suffstats(day_ids, X, y)
    betas = bootstrap(days, XtX, Xty, X, y, day_ids, REPS, SEED)

    ci = np.percentile(betas, [2.5, 97.5], axis=0)
    resid = y - X @ beta_point
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "market": label,
        "n_obs": int(len(sub)),
        "n_calendar_days": int(days.size),
        "beta0": float(beta_point[0]),
        "beta1": float(beta_point[1]),
        "beta2": float(beta_point[2]),
        "beta2_sign": "positive" if beta_point[2] > 0 else ("negative" if beta_point[2] < 0 else "zero"),
        "beta2_ci95": [float(ci[0, 2]), float(ci[1, 2])],
        "beta2_ci95_excludes_zero": bool(ci[0, 2] > 0 or ci[1, 2] < 0),
        "beta0_ci95": [float(ci[0, 0]), float(ci[1, 0])],
        "beta1_ci95": [float(ci[0, 1]), float(ci[1, 1])],
        "r2": r2,
        "bootstrap_reps": REPS,
        "bootstrap_seed": SEED,
        "preregistered_sign_b2": "positive",
        "preregistered_sign_confirmed": bool(beta_point[2] > 0),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from research_core.paths import confirmatory_panels_dir  # noqa: E402

    PAN = confirmatory_panels_dir()
    out = {}
    for label, fname in [("ETH5", "eth5_pre_response_panel.parquet"),
                          ("BASE5", "base5_pre_response_panel.parquet"),
                          ("BASE1", "base1_pre_response_panel.parquet")]:
        print(f"=== {label} ===", flush=True)
        r = run_market(PAN / fname, 86400, label)
        out[label] = r
        print(json.dumps(r, indent=2))
    Path(sys.argv[1] if len(sys.argv) > 1 else "LONG_CONFIRMATORY_RESULTS.json").write_text(json.dumps(out, indent=2))
