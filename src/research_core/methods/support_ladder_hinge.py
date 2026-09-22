"""Hinge-side support-cap robustness ladder (x<=1.5/2/3/5/full).

REWRITE, not a verbatim copy -- see CODE_CORE_EXTRACTION_PLAN.md "REWRITE"
and CODE_CORE_EXTRACTION_AUDIT.md section 6, item 1.

The original producer of this ladder,
work/jfqa_service_confirmatory_long/code/section_e.py, imports its hinge
regression / calendar-day-bootstrap machinery (`design`, `day_cluster_bootstrap`,
`load_swap_sample`) from a module named `diag_common.py` that was loaded via
`sys.path.insert(0, "/private/tmp/.../scratchpad")` -- a different Claude Code
session's ephemeral temp directory. That directory no longer exists on disk
and `diag_common.py` was never committed to the repository, so section_e.py
cannot currently be run as-is, and there is no way to diff this rewrite
against the original byte-for-byte.

This rewrite reconstructs the same hinge design matrix and calendar-day
cluster bootstrap using the pattern from
work/jfqa_service_confirmatory_long/code/run_long_confirmatory_analysis.py
(kept verbatim elsewhere in this repo as scripts/run_confirmation.py), which
is self-contained, verified, and produces the frozen table_t5_confirmation
numbers. The design/bootstrap logic here is intentionally identical to that
script's `design()`/`bootstrap()` functions.

VERIFIED (post-hoc): a frozen copy of the original script's own output,
work/jfqa_service_confirmatory_long/section_e_raw.json (checked into the
source repo despite diag_common.py itself being lost -- see
CODE_CORE_EXTRACTION_AUDIT.md section 6), was located after this rewrite
was written. Re-running this module against the real ETH5 confirmation
panel with reps=500 (the original script's cap-ladder call used
`reps=500`, not the module-level REPS=2000 default -- REPS=2000 only
applies to the primary confirmatory hinge regression, a different call
site) reproduces every field in that frozen JSON byte-for-byte, including
bootstrap CI bounds (which depend on the exact RNG draw sequence, so this
is strong evidence the reconstruction is behaviorally identical to the
original, not just structurally similar). See
tests/integration/test_frozen_values.py::test_live_support_ladder_hinge_matches_recovered_frozen_json.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REPS = 2000
SEED = 20260829

CAPS = (1.5, 2.0, 3.0, 5.0, None)
WINDOWS = ((0.5, 1.5), (0.75, 1.25))


def design(x: np.ndarray) -> np.ndarray:
    """[1, x, (x-1)_+] -- identical to scripts/run_confirmation.py::design."""
    return np.column_stack([np.ones_like(x), x, np.clip(x - 1.0, 0.0, None)])


def _per_day_suffstats(day_ids: np.ndarray, X: np.ndarray, y: np.ndarray):
    days = np.unique(day_ids)
    XtX, Xty = {}, {}
    for d in days:
        m = day_ids == d
        Xd = X[m]
        XtX[d] = Xd.T @ Xd
        Xty[d] = Xd.T @ y[m]
    return days, XtX, Xty


def day_cluster_bootstrap(x: np.ndarray, y: np.ndarray, day_ids: np.ndarray,
                           reps: int = REPS, seed: int = SEED):
    """Calendar-day cluster bootstrap on the hinge design, matching
    scripts/run_confirmation.py::bootstrap exactly (same suffstat-resample
    method, same self-check discipline)."""
    X = design(x)
    days, XtX, Xty = _per_day_suffstats(day_ids, X, y)

    XtX_sum = sum(XtX.values())
    Xty_sum = sum(Xty.values())
    beta_point = np.linalg.solve(XtX_sum, Xty_sum)
    beta_direct, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(beta_point, beta_direct, atol=1e-6), "suffstat aggregation mismatch"

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

    ci = np.percentile(betas, [2.5, 97.5], axis=0)
    return beta_point, ci, betas


def support_cap_ladder(x_all: np.ndarray, y_all: np.ndarray, days_all: np.ndarray,
                        reps: int = REPS, seed: int = SEED, min_n: int = 50) -> dict:
    """x<=1.5/2/3/5/full cap ladder plus the two symmetric windows, matching
    the ORIGINAL section_e.py's cap/window logic (CAPS, WINDOWS values
    verified identical from the audit's direct read of that file)."""
    out_caps: dict = {}
    for cap in CAPS:
        if cap is None:
            m = np.ones_like(x_all, dtype=bool)
            label = "full_support"
        else:
            m = x_all <= cap
            label = f"x<={cap}"
        if m.sum() < min_n:
            out_caps[label] = {"n": int(m.sum()), "note": "too few obs"}
            continue
        beta, ci, _ = day_cluster_bootstrap(x_all[m], y_all[m], days_all[m], reps=reps, seed=seed)
        out_caps[label] = {
            "n": int(m.sum()),
            "beta2": float(beta[2]),
            "beta2_ci95": [float(ci[0, 2]), float(ci[1, 2])],
            "beta2_sign": "positive" if beta[2] > 0 else "negative",
        }

    out_windows: dict = {}
    for lo, hi in WINDOWS:
        m = (x_all >= lo) & (x_all <= hi)
        label = f"{lo}<=x<={hi}"
        if m.sum() < min_n:
            out_windows[label] = {"n": int(m.sum()), "note": "too few obs"}
            continue
        beta, ci, _ = day_cluster_bootstrap(x_all[m], y_all[m], days_all[m], reps=reps, seed=seed)
        out_windows[label] = {
            "n": int(m.sum()),
            "beta2": float(beta[2]),
            "beta2_ci95": [float(ci[0, 2]), float(ci[1, 2])],
            "beta2_sign": "positive" if beta[2] > 0 else "negative",
        }

    return {"support_caps": out_caps, "symmetric_windows": out_windows}
