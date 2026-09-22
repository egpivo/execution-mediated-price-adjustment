#!/usr/bin/env python3
"""Independent semantic, algebra, and harmonized change-point audit.

The script rebuilds S, K, and C from canonical AMM state plus the frozen HF
reference. It does not trust the service-primitives response panels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


BINS = np.asarray([0, .25, .5, .75, 1, 1.25, 1.5, 2, 3, 5, np.inf])
BIN_LABELS = ["[0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1.0)",
              "[1.0,1.25)", "[1.25,1.5)", "[1.5,2.0)", "[2.0,3.0)",
              "[3.0,5.0)", "[5.0,inf)"]
GRID = np.round(np.arange(.30, 2.0001, .05), 2)
SEED = 20260829
REPS = 2000


@dataclass(frozen=True)
class Market:
    name: str
    chain: str
    pool: str
    fee_fraction: float
    canonical: Path
    xstar: Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_market(value: str) -> Market:
    parts = value.split("::")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError("market must be name::chain::pool::fee_fraction::canonical::xstar")
    return Market(parts[0], parts[1], parts[2], float(parts[3]), Path(parts[4]), Path(parts[5]))


def basis(x: np.ndarray, c: float) -> np.ndarray:
    return np.column_stack((np.ones(x.size), x, np.maximum(x - c, 0)))


def ols_sse(x: np.ndarray, y: np.ndarray, c: float) -> float:
    X = basis(x, c)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return float(r @ r)


def point_cp(x: np.ndarray, y: np.ndarray) -> tuple[float, list[float]]:
    sse = [ols_sse(x, y, c) for c in GRID]
    return float(GRID[int(np.argmin(sse))]), sse


def day_stats(x: np.ndarray, y: np.ndarray, day: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique, inv = np.unique(day, return_inverse=True)
    xtx = np.zeros((unique.size, GRID.size, 3, 3))
    xty = np.zeros((unique.size, GRID.size, 3))
    for d in range(unique.size):
        m = inv == d
        xd, yd = x[m], y[m]
        for j, c in enumerate(GRID):
            X = basis(xd, c)
            xtx[d, j] = X.T @ X
            xty[d, j] = X.T @ yd
    return unique, xtx, xty


def bootstrap_cp(x: np.ndarray, y: np.ndarray, day: np.ndarray, reps: int) -> dict:
    unique, xtx, xty = day_stats(x, y, day)
    rng = np.random.default_rng(SEED)
    draws = np.empty(reps)
    for r in range(reps):
        counts = np.bincount(rng.integers(0, unique.size, unique.size), minlength=unique.size).astype(float)
        best_j, best = 0, np.inf
        for j in range(GRID.size):
            A = np.tensordot(counts, xtx[:, j], axes=(0, 0))
            v = np.tensordot(counts, xty[:, j], axes=(0, 0))
            try:
                b = np.linalg.solve(A, v)
            except np.linalg.LinAlgError:
                b = np.linalg.lstsq(A, v, rcond=None)[0]
            # y'Wy is constant across candidates, so minimizing SSE equals
            # minimizing -b'X'Wy.
            score = -float(b @ v)
            if score < best:
                best, best_j = score, j
        draws[r] = GRID[best_j]
    return {
        "mean": float(draws.mean()),
        "p2.5": float(np.percentile(draws, 2.5)),
        "p97.5": float(np.percentile(draws, 97.5)),
        "n_reps": int(reps),
        "edge_share": float(np.mean((draws == GRID[0]) | (draws == GRID[-1]))),
    }


def align_xstar(panel_bn: np.ndarray, xstar: dict) -> dict[str, np.ndarray]:
    ref_bn = np.asarray(xstar["block_number"], dtype=np.int64)
    order = np.argsort(ref_bn, kind="stable")
    sorted_bn = ref_bn[order]
    pos = np.searchsorted(sorted_bn, panel_bn)
    if pos.max(initial=0) >= sorted_bn.size or not np.array_equal(sorted_bn[pos], panel_bn):
        raise ValueError("canonical panel and X* block numbers do not align")
    idx = order[pos]
    return {k: np.asarray(v)[idx] for k, v in xstar.items() if k != "block_number"}


def finite_quantiles(x: np.ndarray) -> dict:
    y = x[np.isfinite(x)]
    return {f"p{int(p*100):02d}": float(np.quantile(y, p)) for p in (.01, .5, .95, .99)} | {
        "min": float(y.min()), "max": float(y.max())
    }


def audit_market(market: Market, reps: int) -> dict:
    canonical_cols = ["block_number", "block_timestamp", "eligible", "n_swaps", "xA_pre", "xA_post"]
    panel = pq.read_table(market.canonical, columns=canonical_cols).to_pydict()
    ref_cols = ["block_number", "xstar_hf_log", "xstar_hf_valid", "max_component_age_s",
                "bybit_age_s", "okx_age_s", "cross_venue_log_dispersion"]
    ref_raw = pq.read_table(market.xstar, columns=ref_cols).to_pydict()
    bn = np.asarray(panel["block_number"], dtype=np.int64)
    ref = align_xstar(bn, ref_raw)

    eligible = np.asarray(panel["eligible"], dtype=bool)
    pre = np.asarray(panel["xA_pre"], dtype=float)
    post = np.asarray(panel["xA_post"], dtype=float)
    S_all = np.asarray(panel["n_swaps"], dtype=np.int64) > 0
    retained = eligible & np.isfinite(pre) & np.isfinite(post) & np.asarray(ref["xstar_hf_valid"], dtype=bool)
    log_ref = np.asarray(ref["xstar_hf_log"], dtype=float)
    gamma = -math.log1p(-market.fee_fraction)
    d_pre = np.abs(log_ref - pre)
    d_post = np.abs(log_ref - post)
    x = d_pre[retained] / gamma
    C = (d_pre - d_post)[retained]
    S = S_all[retained]
    K = d_post[retained] < d_pre[retained]
    ts = np.asarray(panel["block_timestamp"], dtype=np.int64)[retained]
    day = np.asarray([datetime.fromtimestamp(int(v), tz=timezone.utc).date().isoformat() for v in ts])

    no_swap = ~S
    semantic = {
        "no_swap_n": int(no_swap.sum()),
        "no_swap_C_max_abs": float(np.max(np.abs(C[no_swap]))) if no_swap.any() else None,
        "no_swap_C_exact_zero_share": float(np.mean(C[no_swap] == 0)) if no_swap.any() else None,
        "K_implies_S_violations": int(np.sum(K & ~S)),
        "S_and_not_K_n": int(np.sum(S & ~K)),
        "P_S": float(S.mean()),
        "P_K": float(K.mean()),
        "P_K_given_S": float(K[S].mean()),
    }

    rows = []
    for lo, hi, label in zip(BINS[:-1], BINS[1:], BIN_LABELS):
        b = (x >= lo) & (x < hi)
        n = int(b.sum())
        if not n:
            continue
        sb, kb, cb = S[b], K[b], C[b]
        p_s = float(sb.mean())
        c_s = float(cb[sb].mean()) if sb.any() else 0.0
        actual = float(cb.mean())
        product = p_s * c_s
        rows.append({
            "bin": label, "n": n,
            "mean_C": actual,
            "P_S": p_s,
            "E_C_given_S": c_s,
            "P_S_times_E_C_given_S": product,
            "identity_residual": actual - product,
            "P_K": float(kb.mean()),
            "P_K_given_S": float(kb[sb].mean()) if sb.any() else None,
            "E_C_given_K": float(cb[kb].mean()) if kb.any() else None,
        })

    outcomes = {
        "aggregate_correction": (x, C, day),
        "swap_incidence": (x, S.astype(float), day),
        "corrective_incidence": (x, K.astype(float), day),
        "conditional_correction_given_swap": (x[S], C[S], day[S]),
    }
    cp = {}
    for name, (xo, yo, do) in outcomes.items():
        point, sse = point_cp(xo, yo)
        cp[name] = {
            "x_hat": point,
            "bootstrap": bootstrap_cp(xo, yo, do, reps),
            "sse": sse,
            "relative_sse_range": float((max(sse) - min(sse)) / min(sse)),
            "n": int(xo.size),
        }

    all_ts = np.asarray(panel["block_timestamp"], dtype=np.int64)
    spacing = np.diff(all_ts)
    valid_ref = np.asarray(ref["xstar_hf_valid"], dtype=bool)
    max_age = np.asarray(ref["max_component_age_s"], dtype=float)
    return {
        "market": market.name,
        "chain": market.chain,
        "pool": market.pool,
        "fee_fraction": market.fee_fraction,
        "gamma_fee": gamma,
        "canonical_panel_path": str(market.canonical),
        "canonical_panel_sha256": sha256(market.canonical),
        "xstar_path": str(market.xstar),
        "xstar_sha256": sha256(market.xstar),
        "canonical_n": int(bn.size),
        "eligible_n": int(eligible.sum()),
        "reference_valid_n": int(valid_ref.sum()),
        "retained_n": int(retained.sum()),
        "block_cadence_seconds": {"median": float(np.median(spacing)), "p95": float(np.quantile(spacing, .95))},
        "x_support": finite_quantiles(x),
        "reference_max_age_all_quoted_seconds": {
            "median": float(np.nanmedian(max_age)), "p95": float(np.nanquantile(max_age, .95))
        },
        "reference_max_age_retained_seconds": {
            "median": float(np.median(max_age[retained])), "p95": float(np.quantile(max_age[retained], .95))
        },
        "reference_missing_or_stale_n": int((~valid_ref).sum()),
        "reference_missing_or_stale_share": float((~valid_ref).mean()),
        "semantic": semantic,
        "bins": rows,
        "changepoints": cp,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--market", action="append", type=parse_market, required=True)
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    args = p.parse_args()
    if args.reps < 2000:
        raise ValueError("at least 2,000 calendar-day cluster bootstrap replications required")
    results = []
    for market in args.market:
        print(f"auditing {market.name}", flush=True)
        results.append(audit_market(market, args.reps))
    payload = {
        "estimand": {
            "grid": GRID.tolist(), "objective": "unweighted row-level OLS SSE",
            "form": "continuous segmented linear: y=b0+b1*x+b2*(x-c)_+",
            "bootstrap": "calendar-day cluster bootstrap, percentile interval",
            "reps": args.reps, "seed": SEED,
        },
        "markets": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")

    fields = ["market", "chain", "pool", "fee_fraction", "gamma_fee", "canonical_n", "retained_n",
              "block_cadence_median_s", "x_min", "x_p01", "x_p50", "x_p95", "x_p99", "x_max",
              "swap_cp", "swap_ci_low", "swap_ci_high", "corrective_cp", "corrective_ci_low", "corrective_ci_high",
              "conditional_correction_cp", "conditional_correction_ci_low", "conditional_correction_ci_high",
              "aggregate_correction_cp", "aggregate_correction_ci_low", "aggregate_correction_ci_high",
              "reference_source", "reference_clock_type", "reference_median_age_s", "reference_p95_age_s",
              "reference_valid_share"]
    with args.output_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            cps = r["changepoints"]
            def vals(name: str) -> tuple[float, float, float]:
                z = cps[name]
                return z["x_hat"], z["bootstrap"]["p2.5"], z["bootstrap"]["p97.5"]
            scp, slo, shi = vals("swap_incidence")
            kcp, klo, khi = vals("corrective_incidence")
            icp, ilo, ihi = vals("conditional_correction_given_swap")
            acp, alo, ahi = vals("aggregate_correction")
            xs = r["x_support"]
            w.writerow({
                "market": r["market"], "chain": r["chain"], "pool": r["pool"],
                "fee_fraction": r["fee_fraction"], "gamma_fee": r["gamma_fee"],
                "canonical_n": r["canonical_n"], "retained_n": r["retained_n"],
                "block_cadence_median_s": r["block_cadence_seconds"]["median"],
                "x_min": xs["min"], "x_p01": xs["p01"], "x_p50": xs["p50"], "x_p95": xs["p95"], "x_p99": xs["p99"], "x_max": xs["max"],
                "swap_cp": scp, "swap_ci_low": slo, "swap_ci_high": shi,
                "corrective_cp": kcp, "corrective_ci_low": klo, "corrective_ci_high": khi,
                "conditional_correction_cp": icp, "conditional_correction_ci_low": ilo, "conditional_correction_ci_high": ihi,
                "aggregate_correction_cp": acp, "aggregate_correction_ci_low": alo, "aggregate_correction_ci_high": ahi,
                "reference_source": "Bybit ETHUSDC + OKX ETH-USDC direct spot L2",
                "reference_clock_type": "exchange generation time (Bybit cts; OKX ts)",
                "reference_median_age_s": r["reference_max_age_all_quoted_seconds"]["median"],
                "reference_p95_age_s": r["reference_max_age_all_quoted_seconds"]["p95"],
                "reference_valid_share": r["reference_valid_n"] / r["canonical_n"],
            })


if __name__ == "__main__":
    main()
