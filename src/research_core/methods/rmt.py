"""Response-Map Transport (RMT) core estimators.

POST-CONFIRMATORY EXPLORATORY ANALYSIS.

Decomposes window-to-window change in aggregate mean correction into
support (pi), execution-incidence (p), and conditional-magnitude (m)
components. Not confirmatory, structural, or causal.

Frozen objects (do not change here):
  windows H/D, 10 x-bins, S/C/x definitions, calendar-day bootstrap.
"""

import itertools
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Frozen manuscript bins (half-open; last open to +inf).
BIN_EDGES = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf])
BIN_LABELS = [
    "[0,0.25)",
    "[0.25,0.5)",
    "[0.5,0.75)",
    "[0.75,1.0)",
    "[1.0,1.25)",
    "[1.25,1.5)",
    "[1.5,2.0)",
    "[2.0,3.0)",
    "[3.0,5.0)",
    "[5.0,inf)",
]
N_BINS = len(BIN_LABELS)

# Manuscript Table 8 display uses mixed label spellings; normalize for compare.
TABLE8_LABEL_ALIASES = {
    "[0.75,1)": "[0.75,1.0)",
    "[1,1.25)": "[1.0,1.25)",
    "[1.25,1.5)": "[1.25,1.5)",
    "[1.5,2)": "[1.5,2.0)",
    "[2,3)": "[2.0,3.0)",
    "[3,5)": "[3.0,5.0)",
    "[5,inf)": "[5.0,inf)",
}

WINDOWS = {
    "confirmation": ("H", "2026-06-09", "2026-07-05"),
    "discovery": ("D", "2026-07-06", "2026-08-05"),
}

PRIMARY_MARKETS = ("ETH5", "BASE5")
SECONDARY_MARKETS = ("BASE1",)

BP = 1.0e4

# RMT analysis-family seed (no prior RMT-specific seed existed).
RMT_SEED = 20260907
RMT_REPS = 2000
INVALID_SHARE_GATE = 0.05


@dataclass(frozen=True)
class BinMaps:
    """Frozen-bin response-map objects for one (market, window)."""

    n: np.ndarray  # (10,)
    n_swap: np.ndarray
    pi: np.ndarray
    p: np.ndarray
    m: np.ndarray  # native log units; NaN if undefined
    r: np.ndarray  # p * m when defined, else NaN
    mean_c: float
    mean_c_from_bins: float
    p_defined: np.ndarray  # bool
    m_defined: np.ndarray  # bool


def assign_bins(x: np.ndarray) -> np.ndarray:
    """Return bin index in {0..9}; values outside [0, inf) are -1."""
    # np.digitize with right=False: edges[i-1] <= x < edges[i] for i=1..len-1
    idx = np.digitize(x, BIN_EDGES[1:-1], right=False)
    bad = ~np.isfinite(x) | (x < 0)
    out = idx.astype(np.int64)
    out[bad] = -1
    return out


def maps_from_counts(
    n: np.ndarray,
    n_swap: np.ndarray,
    sum_c_swap: np.ndarray,
) -> BinMaps:
    """Build BinMaps from per-bin counts (native log units for C)."""
    n = np.asarray(n, dtype=np.float64)
    n_swap = np.asarray(n_swap, dtype=np.float64)
    sum_c_swap = np.asarray(sum_c_swap, dtype=np.float64)
    if n.shape != (N_BINS,) or n_swap.shape != (N_BINS,) or sum_c_swap.shape != (N_BINS,):
        raise ValueError("expected shape (10,) for bin arrays")

    total = float(n.sum())
    if total <= 0:
        raise ValueError("empty sample")

    pi = n / total
    p = np.full(N_BINS, np.nan)
    m = np.full(N_BINS, np.nan)
    p_defined = n > 0
    m_defined = n_swap > 0
    p[p_defined] = n_swap[p_defined] / n[p_defined]
    m[m_defined] = sum_c_swap[m_defined] / n_swap[m_defined]
    r = np.full(N_BINS, np.nan)
    both = p_defined & m_defined
    r[both] = p[both] * m[both]
    # bins with n>0 but n_swap==0: E[C|x]=0 and product=0
    zero_swap = p_defined & ~m_defined
    r[zero_swap] = 0.0

    # Aggregate mean C = sum_b pi * (n_swap/n) * m = sum sum_c_swap / total
    # when every swap-bin with mass is defined; also equals total swapped C / N
    # because C=0 when S=0.
    mean_c = float(sum_c_swap.sum() / total)
    if np.all(p_defined) and np.all(np.isfinite(r[p_defined])):
        mean_c_from_bins = float(np.nansum(pi * np.nan_to_num(r, nan=0.0)))
    else:
        mean_c_from_bins = float("nan")

    return BinMaps(
        n=n.astype(np.int64),
        n_swap=n_swap.astype(np.int64),
        pi=pi,
        p=p,
        m=m,
        r=r,
        mean_c=mean_c,
        mean_c_from_bins=mean_c_from_bins,
        p_defined=p_defined,
        m_defined=m_defined,
    )


def aggregate_day_stats(
    n_by_day: np.ndarray,
    n_swap_by_day: np.ndarray,
    sum_c_swap_by_day: np.ndarray,
    day_counts: Optional[np.ndarray] = None,
) -> BinMaps:
    """Aggregate (n_days, 10) day stats with optional bootstrap multiplicities."""
    if day_counts is None:
        n = n_by_day.sum(axis=0)
        n_swap = n_swap_by_day.sum(axis=0)
        sum_c = sum_c_swap_by_day.sum(axis=0)
    else:
        w = np.asarray(day_counts, dtype=np.float64)
        n = w @ n_by_day
        n_swap = w @ n_swap_by_day
        sum_c = w @ sum_c_swap_by_day
    return maps_from_counts(n, n_swap, sum_c)


def build_day_bin_stats(
    x: np.ndarray,
    has_swap: np.ndarray,
    c: np.ndarray,
    day_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unique_days and (n_days,10) arrays: n, n_swap, sum_c_swap."""
    x = np.asarray(x, dtype=np.float64)
    has_swap = np.asarray(has_swap, dtype=bool)
    c = np.asarray(c, dtype=np.float64)
    day_ids = np.asarray(day_ids)
    bins = assign_bins(x)
    ok = bins >= 0
    unique_days, inv = np.unique(day_ids[ok], return_inverse=True)
    n_days = unique_days.size
    n = np.zeros((n_days, N_BINS), dtype=np.float64)
    n_swap = np.zeros((n_days, N_BINS), dtype=np.float64)
    sum_c_swap = np.zeros((n_days, N_BINS), dtype=np.float64)

    b = bins[ok]
    d = inv
    s = has_swap[ok]
    cc = c[ok]
    for i in range(d.size):
        di = d[i]
        bi = b[i]
        n[di, bi] += 1.0
        if s[i]:
            n_swap[di, bi] += 1.0
            sum_c_swap[di, bi] += cc[i]
    return unique_days, n, n_swap, sum_c_swap


def F(pi: np.ndarray, p: np.ndarray, m: np.ndarray) -> float:
    """Aggregate mean correction under (pi, p, m); native log units."""
    return float(np.sum(pi * p * m))


def require_finite_maps(maps_D: BinMaps, maps_H: BinMaps) -> None:
    """Fail closed before transport/Shapley if any bin's p or m is
    non-finite (e.g. a zero-swap bin, where n_swap==0 leaves m undefined =
    NaN). Without this, F(pi,p,m) = sum(pi*p*m) silently becomes NaN, and a
    naive `abs(residual) > tol` gate is fail-OPEN on NaN (`abs(nan) > x` is
    False in Python) -- exactly the failure mode this function exists to
    catch (CODE_REVIEW.md repair #9).

    Raises ValueError with a message identifying which side/array failed;
    callers decide whether that becomes a SystemExit, RuntimeError, etc.
    """
    for label, maps in (("D", maps_D), ("H", maps_H)):
        for name, arr in (("p", maps.p), ("m", maps.m)):
            if not np.isfinite(arr).all():
                bad_bins = np.flatnonzero(~np.isfinite(arr)).tolist()
                raise ValueError(
                    f"non-finite {name} in window {label} at bin(s) {bad_bins} "
                    f"(likely a zero-swap bin) -- refusing to compute transport/Shapley "
                    f"over an undefined map"
                )


def require_finite_residual(residual: float, *, tol: float, label: str = "") -> None:
    """Fail closed on a non-finite residual before checking tolerance --
    `abs(nan) > tol` is False in Python, so a naive tolerance check alone is
    fail-open on NaN (CODE_REVIEW.md repair #9)."""
    if not np.isfinite(residual):
        raise ValueError(f"non-finite residual{f' ({label})' if label else ''}: {residual!r}")
    if abs(residual) > tol:
        raise ValueError(f"residual{f' ({label})' if label else ''} {residual} exceeds tolerance {tol}")


def transport_error(pi_H: np.ndarray, maps_D: BinMaps, maps_H: BinMaps) -> dict:
    """Analysis A: Discovery map on Confirmation support."""
    c_h_from_d = F(pi_H, maps_D.p, maps_D.m)
    c_h = F(maps_H.pi, maps_H.p, maps_H.m)
    te = c_h - c_h_from_d
    return {
        "C_H": c_h,
        "C_H_from_D": c_h_from_d,
        "TE_D_to_H": te,
        "C_H_bp": c_h * BP,
        "C_H_from_D_bp": c_h_from_d * BP,
        "TE_D_to_H_bp": te * BP,
    }


def standardized_difference(maps_D: BinMaps, maps_H: BinMaps) -> dict:
    """Analysis B: common-support standardization with pi_bar = 0.5*(pi_D+pi_H)."""
    pi_bar = 0.5 * (maps_D.pi + maps_H.pi)
    c_d_std = F(pi_bar, maps_D.p, maps_D.m)
    c_h_std = F(pi_bar, maps_H.p, maps_H.m)
    diff = c_h_std - c_d_std
    return {
        "pi_bar": pi_bar,
        "C_D_std": c_d_std,
        "C_H_std": c_h_std,
        "STD_DIFF": diff,
        "C_D_std_bp": c_d_std * BP,
        "C_H_std_bp": c_h_std * BP,
        "STD_DIFF_bp": diff * BP,
        "pi_bar_sum": float(pi_bar.sum()),
    }


def shapley_decomposition(maps_D: BinMaps, maps_H: BinMaps) -> dict:
    """Exact 3-factor Shapley decomposition of DELTA = F_H - F_D.

    Factors: pi (support), p (incidence), m (magnitude).
    """
    start = {"pi": maps_D.pi, "p": maps_D.p, "m": maps_D.m}
    end = {"pi": maps_H.pi, "p": maps_H.p, "m": maps_H.m}
    factors = ("pi", "p", "m")
    f_d = F(start["pi"], start["p"], start["m"])
    f_h = F(end["pi"], end["p"], end["m"])
    delta = f_h - f_d

    contrib = {k: 0.0 for k in factors}
    n_perm = 0
    for order in itertools.permutations(factors):
        n_perm += 1
        cur = dict(start)
        f_cur = F(cur["pi"], cur["p"], cur["m"])
        for fac in order:
            cur[fac] = end[fac]
            f_new = F(cur["pi"], cur["p"], cur["m"])
            contrib[fac] += f_new - f_cur
            f_cur = f_new

    for k in factors:
        contrib[k] /= float(n_perm)

    residual = delta - (contrib["pi"] + contrib["p"] + contrib["m"])
    return {
        "DELTA": delta,
        "DELTA_pi": contrib["pi"],
        "DELTA_p": contrib["p"],
        "DELTA_m": contrib["m"],
        "DELTA_bp": delta * BP,
        "DELTA_pi_bp": contrib["pi"] * BP,
        "DELTA_p_bp": contrib["p"] * BP,
        "DELTA_m_bp": contrib["m"] * BP,
        "shapley_residual": residual,
        "n_permutations": n_perm,
        "F_D": f_d,
        "F_H": f_h,
    }


def moore_penrose_wald(d: np.ndarray, cov: np.ndarray, rcond: float = 1e-10) -> dict:
    """Generalized Wald T = d' Sigma^+ d with rank / condition diagnostics."""
    d = np.asarray(d, dtype=np.float64).reshape(-1)
    cov = np.asarray(cov, dtype=np.float64)
    # Numerical symmetrization
    cov = 0.5 * (cov + cov.T)
    w, v = np.linalg.eigh(cov)
    # Relative threshold on eigenvalues
    cutoff = rcond * float(np.max(np.abs(w))) if w.size else 0.0
    keep = w > cutoff
    rank = int(keep.sum())
    inv_w = np.zeros_like(w)
    inv_w[keep] = 1.0 / w[keep]
    pinv = (v * inv_w) @ v.T
    t = float(d @ pinv @ d)
    cond = float(np.max(w[keep]) / np.min(w[keep])) if rank > 0 else float("inf")
    return {
        "T": t,
        "rank": rank,
        "eig_min_kept": float(np.min(w[keep])) if rank else float("nan"),
        "eig_max": float(np.max(w)) if w.size else float("nan"),
        "condition_kept": cond,
        "n_zero_eig_dropped": int((~keep).sum()),
        "pinv": pinv,
    }


def bootstrap_p_value(t_obs: float, t_star: np.ndarray) -> float:
    t_star = np.asarray(t_star, dtype=np.float64)
    t_star = t_star[np.isfinite(t_star)]
    if t_star.size == 0 or not np.isfinite(t_obs):
        return float("nan")
    return float(np.mean(t_star >= t_obs))


def percentile_ci(draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    draws = np.asarray(draws, dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float("nan"), float("nan")
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)
    return float(np.percentile(draws, lo)), float(np.percentile(draws, hi))


def invalid_share(n_invalid: int, n_reps: int) -> float:
    return float(n_invalid) / float(n_reps) if n_reps else float("nan")
