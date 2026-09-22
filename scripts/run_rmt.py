#!/usr/bin/env python3
"""Run frozen Response-Map Transport (RMT) analysis.

POST-CONFIRMATORY EXPLORATORY ANALYSIS.
Does not modify manuscript, confirmation hinge results, or CME work.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# rmt_core.py now lives at src/research_core/methods/rmt.py (renamed for
# package hygiene, see CODE_CORE_EXTRACTION_PLAN.md KEEP table).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "research_core" / "methods"))

from rmt import (  # noqa: E402
    BIN_LABELS,
    BP,
    INVALID_SHARE_GATE,
    N_BINS,
    PRIMARY_MARKETS,
    RMT_REPS,
    RMT_SEED,
    SECONDARY_MARKETS,
    TABLE8_LABEL_ALIASES,
    aggregate_day_stats,
    bootstrap_p_value,
    build_day_bin_stats,
    invalid_share,
    maps_from_counts,
    moore_penrose_wald,
    percentile_ci,
    require_finite_maps,
    require_finite_residual,
    shapley_decomposition,
    standardized_difference,
    transport_error,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core.paths import (  # noqa: E402
    confirmatory_panels_dir,
    hf_response_full_derived_dir,
    service_primitives_derived_dir,
)

WORK = Path(__file__).resolve().parents[1]  # code/
PROJECT = WORK.parent  # cfmm-clob-price-tracking/ (manuscript lives one level up from code)
DERIVED = WORK / "outputs" / "tables" / "derived"
FIGURES = WORK / "outputs" / "tables" / "figures"
DERIVED.mkdir(parents=True, exist_ok=True)
MANUSCRIPT_TABLE8 = PROJECT / "manuscript" / "generated" / "table_a_decomposition_bins.tex"

# Canonical response panels. All paths resolve through research_core.paths
# (CODE_REVIEW.md repair #5) -- set RESEARCH_CORE_EXTERNAL_CACHE to a root
# that preserves the original jfqa_*/... relative layout to use these.
RESPONSE_PANELS = {
    ("ETH5", "discovery"): hf_response_full_derived_dir() / "eth5_response_panel_hf_v1.parquet",
    ("BASE5", "discovery"): service_primitives_derived_dir() / "base5_response_panel_hf_v1.parquet",
    ("BASE1", "discovery"): service_primitives_derived_dir() / "base1_response_panel_hf_v1.parquet",
    ("ETH5", "confirmation"): confirmatory_panels_dir() / "eth5_pre_response_panel.parquet",
    ("BASE5", "confirmation"): confirmatory_panels_dir() / "base5_pre_response_panel.parquet",
    ("BASE1", "confirmation"): confirmatory_panels_dir() / "base1_pre_response_panel.parquet",
}

# Manuscript Table 8 rounded targets (bp for m and r; p unitless). Used for gate.
TABLE8_TOL_P = 0.0015  # display is 3 decimals
TABLE8_TOL_BP = 0.0015
TABLE8_TOL_N = 0  # exact integer match expected


def load_retained(market: str, window: str):
    path = RESPONSE_PANELS[(market, window)]
    cols = ["block_timestamp", "retained_hf", "has_focal_swap", "x", "correction_hf"]
    t = pq.read_table(path, columns=cols).to_pandas()
    r = t[t["retained_hf"]].copy()
    r = r[np.isfinite(r["x"]) & np.isfinite(r["correction_hf"])]
    # Calendar day in UTC seconds (matches existing confirmatory convention).
    r["day"] = (r["block_timestamp"].astype("int64") // 86400).astype("int64")
    return r, path


def compute_point_maps(df):
    unique_days, n, n_swap, sum_c = build_day_bin_stats(
        df["x"].to_numpy(),
        df["has_focal_swap"].to_numpy(),
        df["correction_hf"].to_numpy(),
        df["day"].to_numpy(),
    )
    maps = aggregate_day_stats(n, n_swap, sum_c)
    return unique_days, n, n_swap, sum_c, maps


# Expected value-column order in the manuscript's full decomposition table
# (manuscript/generated/table_a_decomposition_bins.tex), after the
# Market/window and bin-label columns: N, N_S, P(S|x), E[C|S,x] (bp),
# E[C|x] (bp), Product (bp), Residual. N_S was added to expose exact
# swap-block counts (round-2 cold-review closure); this parser must be
# kept in sync with work/irfa_manuscript_rebuild/code/build_paper_assets.py's
# t4_full row layout. If that generator's column order ever changes again,
# this constant -- and only this constant -- should need updating.
TABLE8_VALUE_COLUMNS = ("n", "n_swap", "p", "m_bp", "mean_bp", "product_bp", "residual")
TABLE8_EXPECTED_NCOLS = len(TABLE8_VALUE_COLUMNS)


def parse_table8(tex_path: Path) -> dict:
    """Parse the manuscript's full bin decomposition table into
    {(market, window, bin_label): dict}, reading the exact stored integer
    N_S column rather than deriving it from a rounded displayed
    probability. Fails closed (raises) on any schema drift: wrong column
    count, non-integer N/N_S, N_S > N, or a bin whose displayed P(S|x)
    materially disagrees with N_S/N."""
    text = tex_path.read_text()
    rows = {}
    current_mw = None
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line.startswith("ETH") and not line.startswith("BASE") and not line.startswith("&"):
            continue
        parts = [p.strip() for p in line.replace("\\\\", "").split("&")]
        if line.startswith("ETH") or line.startswith("BASE"):
            # ETH5 Discovery & [0,0.25) & ...
            head = parts[0]
            market, window_word = head.split()
            window = "discovery" if window_word.lower().startswith("disc") else "confirmation"
            current_mw = (market, window)
            bin_lab = TABLE8_LABEL_ALIASES.get(parts[1], parts[1])
            vals = parts[2:]
        else:
            if current_mw is None:
                continue
            bin_lab = TABLE8_LABEL_ALIASES.get(parts[1], parts[1])
            vals = parts[2:]
        # \addlinespace and other non-data continuation lines parse to a
        # short or empty vals list; skip them rather than silently
        # accepting a malformed row.
        if len(vals) == 0:
            continue
        if len(vals) != TABLE8_EXPECTED_NCOLS:
            raise ValueError(
                f"table8 schema drift at {tex_path.name}:{lineno}: expected "
                f"{TABLE8_EXPECTED_NCOLS} value columns {TABLE8_VALUE_COLUMNS}, "
                f"found {len(vals)}: {vals!r}. The generator "
                f"(work/irfa_manuscript_rebuild/code/build_paper_assets.py) and "
                f"this parser's TABLE8_VALUE_COLUMNS have gone out of sync."
            )
        try:
            n = int(vals[0].replace(",", ""))
            n_swap = int(vals[1].replace(",", ""))
        except ValueError as e:
            raise ValueError(
                f"table8 integer-parse failure at {tex_path.name}:{lineno} "
                f"for {current_mw} {bin_lab}: N={vals[0]!r} N_S={vals[1]!r}"
            ) from e
        p = float(vals[2])
        m_bp = float(vals[3])
        mean_bp = float(vals[4])
        product_bp = float(vals[5])
        residual = float(vals[6])
        if n_swap > n:
            raise ValueError(
                f"table8 N_S > N at {tex_path.name}:{lineno} for {current_mw} "
                f"{bin_lab}: N_S={n_swap} > N={n}"
            )
        if n > 0:
            implied_p = n_swap / n
            # Display rounds P(S|x) to 3 decimals; a real N_S/N vs. displayed
            # p disagreement beyond that rounding indicates the table's N,
            # N_S, and P(S|x) columns were generated from inconsistent
            # sources.
            if abs(implied_p - p) > TABLE8_TOL_P:
                raise ValueError(
                    f"table8 N_S/N disagrees with displayed P(S|x) at "
                    f"{tex_path.name}:{lineno} for {current_mw} {bin_lab}: "
                    f"N_S/N={implied_p:.6f} vs displayed p={p}"
                )
        rows[(current_mw[0], current_mw[1], bin_lab)] = {
            "n": n,
            "n_swap": n_swap,
            "p": p,
            "m_bp": m_bp,
            "mean_bp": mean_bp,
            "product_bp": product_bp,
            "residual": residual,
        }
    return rows


def verify_table8(market: str, window: str, maps, table8: dict) -> list:
    """Return list of discrepancy strings; empty means pass. Checks N and
    N_S exactly (integer equality); p, m, and the product (r = p*m,
    equivalently E[C|x] by the S=0-implies-C=0 identity) within
    display-rounding tolerance."""
    issues = []
    for i, lab in enumerate(BIN_LABELS):
        key = (market, window, lab)
        if key not in table8:
            issues.append(f"missing Table8 key {key}")
            continue
        t = table8[key]
        if abs(int(maps.n[i]) - t["n"]) > TABLE8_TOL_N:
            issues.append(f"{key} n {maps.n[i]} vs {t['n']}")
        if abs(int(maps.n_swap[i]) - t["n_swap"]) > TABLE8_TOL_N:
            issues.append(f"{key} n_swap {maps.n_swap[i]} vs {t['n_swap']}")
        if abs(float(maps.p[i]) - t["p"]) > TABLE8_TOL_P:
            issues.append(f"{key} p {maps.p[i]:.6f} vs {t['p']}")
        m_bp = float(maps.m[i]) * BP
        r_bp = float(maps.r[i]) * BP
        if abs(m_bp - t["m_bp"]) > TABLE8_TOL_BP:
            issues.append(f"{key} m_bp {m_bp:.6f} vs {t['m_bp']}")
        if abs(r_bp - t["product_bp"]) > TABLE8_TOL_BP:
            issues.append(f"{key} product_bp {r_bp:.6f} vs {t['product_bp']}")
    return issues


def classify_pattern(point: dict, boot: dict) -> str:
    """Descriptive exploratory pattern; not a formal decision rule."""
    # Inference instability gate
    for name in ("TE_D_to_H", "STD_DIFF", "DELTA_pi", "DELTA_p", "DELTA_m", "H0_ext", "H0_int"):
        share = boot["invalid_share"].get(name, 0.0)
        if share > INVALID_SHARE_GATE:
            return "INFERENCE_UNSTABLE"

    comps = {
        "SUPPORT_SHIFT_DOMINANT": abs(point["DELTA_pi_bp"]),
        "INCIDENCE_SHIFT_DOMINANT": abs(point["DELTA_p_bp"]),
        "MAGNITUDE_SHIFT_DOMINANT": abs(point["DELTA_m_bp"]),
    }
    total = sum(comps.values())
    te = abs(point["TE_D_to_H_bp"])
    std = abs(point["STD_DIFF_bp"])
    delta = abs(point["DELTA_bp"])

    # Little difference: small absolute transport objects relative to mean levels
    scale = max(abs(point["mean_c_D_bp"]), abs(point["mean_c_H_bp"]), 1e-9)
    if max(te, std, delta) < 0.05 * scale and total < 0.05 * scale:
        # Also require non-rejection of both omnibus tests if available
        p_ext = boot["p_values"].get("H0_ext")
        p_int = boot["p_values"].get("H0_int")
        if (p_ext is None or p_ext > 0.10) and (p_int is None or p_int > 0.10):
            return "LITTLE_TRANSPORT_DIFFERENCE"

    # Dominant component: largest |Shapley| share >= 60% and clearly larger than others
    if total > 0:
        ranked = sorted(comps.items(), key=lambda kv: kv[1], reverse=True)
        top_name, top_val = ranked[0]
        second = ranked[1][1]
        if top_val / total >= 0.60 and top_val >= 1.5 * second:
            return top_name
        return "MULTIPLE_COMPONENTS_MATERIAL"
    return "LITTLE_TRANSPORT_DIFFERENCE"


def run_bootstrap_market(
    days_D,
    n_D,
    nsw_D,
    sc_D,
    days_H,
    n_H,
    nsw_H,
    sc_H,
    maps_D,
    maps_H,
    reps: int,
    seed: int,
):
    """Calendar-day bootstrap within each window; shared replicates for all stats."""
    rng = np.random.default_rng(seed)
    n_days_d = days_D.size
    n_days_h = days_H.size

    te_draws = np.full(reps, np.nan)
    std_draws = np.full(reps, np.nan)
    dpi_draws = np.full(reps, np.nan)
    dp_draws = np.full(reps, np.nan)
    dm_draws = np.full(reps, np.nan)
    dext_draws = np.full((reps, N_BINS), np.nan)
    dint_draws = np.full((reps, N_BINS), np.nan)

    inv = {
        "TE_D_to_H": 0,
        "STD_DIFF": 0,
        "DELTA_pi": 0,
        "DELTA_p": 0,
        "DELTA_m": 0,
        "H0_ext": 0,
        "H0_int": 0,
        "any_maps": 0,
    }

    for r in range(reps):
        counts_d = np.bincount(rng.integers(0, n_days_d, size=n_days_d), minlength=n_days_d).astype(
            np.float64
        )
        counts_h = np.bincount(rng.integers(0, n_days_h, size=n_days_h), minlength=n_days_h).astype(
            np.float64
        )
        md = aggregate_day_stats(n_D, nsw_D, sc_D, counts_d)
        mh = aggregate_day_stats(n_H, nsw_H, sc_H, counts_h)

        # Extensives need all p defined
        if not (np.all(md.p_defined) and np.all(mh.p_defined)):
            inv["H0_ext"] += 1
            inv["any_maps"] += 1
        else:
            dext_draws[r] = mh.p - md.p

        # Intensives need all m defined
        if not (np.all(md.m_defined) and np.all(mh.m_defined)):
            inv["H0_int"] += 1
            # TE uses m_D on all bins with pi_H>0; Shapley/STD need both m maps.
            inv["TE_D_to_H"] += 1
            inv["STD_DIFF"] += 1
            inv["DELTA_pi"] += 1
            inv["DELTA_p"] += 1
            inv["DELTA_m"] += 1
            inv["any_maps"] += 1
            continue

        # At this point m defined everywhere; p may still be undefined if n=0
        if not (np.all(md.p_defined) and np.all(mh.p_defined)):
            inv["TE_D_to_H"] += 1
            inv["STD_DIFF"] += 1
            inv["DELTA_pi"] += 1
            inv["DELTA_p"] += 1
            inv["DELTA_m"] += 1
            inv["any_maps"] += 1
            continue

        dint_draws[r] = mh.m - md.m
        te = transport_error(mh.pi, md, mh)
        std = standardized_difference(md, mh)
        sh = shapley_decomposition(md, mh)
        # p_defined/m_defined are already checked above, so this residual
        # should be finite by construction; guard explicitly anyway (repair
        # #9 -- fail-closed on non-finite, not just out-of-tolerance).
        require_finite_residual(sh["shapley_residual"], tol=1e-9, label="bootstrap Shapley")

        te_draws[r] = te["TE_D_to_H"]
        std_draws[r] = std["STD_DIFF"]
        dpi_draws[r] = sh["DELTA_pi"]
        dp_draws[r] = sh["DELTA_p"]
        dm_draws[r] = sh["DELTA_m"]

    # Observed vectors
    d_ext_obs = maps_H.p - maps_D.p
    d_int_obs = maps_H.m - maps_D.m

    def cov_from_draws(draws):
        valid = np.isfinite(draws).all(axis=1)
        X = draws[valid]
        if X.shape[0] < 2:
            return np.full((N_BINS, N_BINS), np.nan), 0
        # Sample covariance of difference vectors
        return np.cov(X, rowvar=False, ddof=1), int(X.shape[0])

    cov_ext, n_ext = cov_from_draws(dext_draws)
    cov_int, n_int = cov_from_draws(dint_draws)

    wald_ext = moore_penrose_wald(d_ext_obs, cov_ext) if np.isfinite(cov_ext).all() else None
    wald_int = moore_penrose_wald(d_int_obs, cov_int) if np.isfinite(cov_int).all() else None

    def p_from_wald(wald, draws, d_obs):
        if wald is None:
            return float("nan"), np.array([])
        pinv = wald["pinv"]
        valid = np.isfinite(draws).all(axis=1)
        centered = draws[valid] - d_obs.reshape(1, -1)
        t_star = np.einsum("ij,jk,ik->i", centered, pinv, centered)
        return bootstrap_p_value(wald["T"], t_star), t_star

    p_ext, _ = p_from_wald(wald_ext, dext_draws, d_ext_obs)
    p_int, _ = p_from_wald(wald_int, dint_draws, d_int_obs)

    def ci_pack(name, draws):
        lo, hi = percentile_ci(draws)
        share = invalid_share(inv[name], reps)
        return {
            "mean": float(np.nanmean(draws)),
            "ci_lo": lo,
            "ci_hi": hi,
            "n_valid": int(np.isfinite(draws).sum()),
            "n_invalid": inv[name],
            "invalid_share": share,
            "BOOTSTRAP_SUPPORT_INSTABILITY": share > INVALID_SHARE_GATE,
        }

    return {
        "invalid_counts": inv,
        "invalid_share": {k: invalid_share(v, reps) for k, v in inv.items()},
        "ci": {
            "TE_D_to_H": ci_pack("TE_D_to_H", te_draws),
            "STD_DIFF": ci_pack("STD_DIFF", std_draws),
            "DELTA_pi": ci_pack("DELTA_pi", dpi_draws),
            "DELTA_p": ci_pack("DELTA_p", dp_draws),
            "DELTA_m": ci_pack("DELTA_m", dm_draws),
        },
        "p_values": {"H0_ext": p_ext, "H0_int": p_int},
        "wald": {
            "H0_ext": None
            if wald_ext is None
            else {k: wald_ext[k] for k in ("T", "rank", "condition_kept", "eig_min_kept", "eig_max", "n_zero_eig_dropped")},
            "H0_int": None
            if wald_int is None
            else {k: wald_int[k] for k in ("T", "rank", "condition_kept", "eig_min_kept", "eig_max", "n_zero_eig_dropped")},
        },
        "n_cov_ext": n_ext,
        "n_cov_int": n_int,
        "draws_bp": {
            "TE_D_to_H": te_draws * BP,
            "STD_DIFF": std_draws * BP,
            "DELTA_pi": dpi_draws * BP,
            "DELTA_p": dp_draws * BP,
            "DELTA_m": dm_draws * BP,
        },
    }


def make_figure(market_results: dict, out_path: Path):
    """Legacy analysis-time figure hook.

    Manuscript figures are style-normalized from frozen
    ``derived/rmt_results.json`` via
    ``work/figure_style_normalization/render_all_figures.py`` (no re-estimation).
    This helper retains analysis-run convenience output only.
    """
    import sys
    from pathlib import Path as _Path

    style_dir = _Path(__file__).resolve().parents[2] / "figure_style_normalization"
    sys.path.insert(0, str(style_dir))
    # Prefer stored-artifact render when rmt_results.json already exists.
    results_path = _Path(__file__).resolve().parents[1] / "derived" / "rmt_results.json"
    if results_path.exists():
        from render_all_figures import render_rmt

        render_rmt()
        return
    raise SystemExit(
        "STOP: make_figure requires frozen derived/rmt_results.json for "
        "style-normalized rendering; do not invent a parallel plot path."
    )


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    table8 = parse_table8(MANUSCRIPT_TABLE8)
    if not table8:
        raise SystemExit("STOP: failed to parse Table 8")

    markets = list(PRIMARY_MARKETS)  # BASE1 optional; skip unless trivial later
    market_results = {}
    stop_flags = []

    for market in markets:
        print(f"=== {market} point maps ===", flush=True)
        df_d, path_d = load_retained(market, "discovery")
        df_h, path_h = load_retained(market, "confirmation")
        days_d, n_d, nsw_d, sc_d, maps_d = compute_point_maps(df_d)
        days_h, n_h, nsw_h, sc_h, maps_h = compute_point_maps(df_h)

        # Identity check
        if abs(maps_d.mean_c - maps_d.mean_c_from_bins) > 1e-12:
            stop_flags.append(f"{market} D identity fail")
        if abs(maps_h.mean_c - maps_h.mean_c_from_bins) > 1e-12:
            stop_flags.append(f"{market} H identity fail")

        issues_d = verify_table8(market, "discovery", maps_d, table8)
        issues_h = verify_table8(market, "confirmation", maps_h, table8)
        if issues_d or issues_h:
            print("TABLE8 MISMATCH:", issues_d[:5], issues_h[:5])
            stop_flags.append(f"{market} Table8 mismatch")
            # STOP per spec — still record diagnostics but do not proceed to RMT headline
            market_results[market] = {
                "stop": True,
                "issues_d": issues_d,
                "issues_h": issues_h,
                "path_d": str(path_d),
                "path_h": str(path_h),
            }
            continue

        # CODE_REVIEW.md repair #9: fail closed on non-finite p/m before the
        # Shapley call (matches support_ladder.py::evaluate's guard).
        try:
            require_finite_maps(maps_d, maps_h)
        except ValueError as exc:
            raise SystemExit(f"STOP: {market}: {exc}") from exc

        te = transport_error(maps_h.pi, maps_d, maps_h)
        std = standardized_difference(maps_d, maps_h)
        sh = shapley_decomposition(maps_d, maps_h)
        try:
            require_finite_residual(sh["shapley_residual"], tol=1e-12, label=f"{market} Shapley")
        except ValueError as exc:
            raise SystemExit(f"STOP: {exc}") from exc

        point = {
            "mean_c_D": maps_d.mean_c,
            "mean_c_H": maps_h.mean_c,
            "mean_c_D_bp": maps_d.mean_c * BP,
            "mean_c_H_bp": maps_h.mean_c * BP,
            **te,
            **{k: std[k] for k in std if k != "pi_bar"},
            **{k: sh[k] for k in sh if k != "pi_bar"},
            "n_D": int(maps_d.n.sum()),
            "n_H": int(maps_h.n.sum()),
            "n_days_D": int(days_d.size),
            "n_days_H": int(days_h.size),
        }

        print(f"=== {market} bootstrap reps={RMT_REPS} seed={RMT_SEED} ===", flush=True)
        boot = run_bootstrap_market(
            days_d, n_d, nsw_d, sc_d, days_h, n_h, nsw_h, sc_h, maps_d, maps_h, RMT_REPS, RMT_SEED
        )

        # Convert CIs to bp in a dedicated view
        ci_bp = {}
        for name, pack in boot["ci"].items():
            lo, hi = percentile_ci(boot["draws_bp"][name])
            share = pack["invalid_share"]
            ci_bp[name] = {
                "ci_lo_bp": lo,
                "ci_hi_bp": hi,
                "invalid_share": share,
                "BOOTSTRAP_SUPPORT_INSTABILITY": share > INVALID_SHARE_GATE,
                "n_valid": pack["n_valid"],
            }

        pattern = classify_pattern(point, boot)
        market_results[market] = {
            "stop": False,
            "path_d": str(path_d),
            "path_h": str(path_h),
            "maps_D": maps_d,
            "maps_H": maps_h,
            "point": point,
            "boot": boot,
            "ci_bp": ci_bp,
            "pattern": pattern,
            "bin_rows": [
                {
                    "bin": BIN_LABELS[i],
                    "pi_D": float(maps_d.pi[i]),
                    "pi_H": float(maps_h.pi[i]),
                    "p_D": float(maps_d.p[i]),
                    "p_H": float(maps_h.p[i]),
                    "m_D_bp": float(maps_d.m[i] * BP),
                    "m_H_bp": float(maps_h.m[i] * BP),
                    "r_D_bp": float(maps_d.r[i] * BP),
                    "r_H_bp": float(maps_h.r[i] * BP),
                    "n_D": int(maps_d.n[i]),
                    "n_H": int(maps_h.n[i]),
                    "n_swap_D": int(maps_d.n_swap[i]),
                    "n_swap_H": int(maps_h.n_swap[i]),
                }
                for i in range(N_BINS)
            ],
        }
        print(
            json.dumps(
                {
                    "market": market,
                    "TE_bp": point["TE_D_to_H_bp"],
                    "STD_DIFF_bp": point["STD_DIFF_bp"],
                    "DELTA_pi_bp": point["DELTA_pi_bp"],
                    "DELTA_p_bp": point["DELTA_p_bp"],
                    "DELTA_m_bp": point["DELTA_m_bp"],
                    "p_ext": boot["p_values"]["H0_ext"],
                    "p_int": boot["p_values"]["H0_int"],
                    "pattern": pattern,
                    "invalid_share": boot["invalid_share"],
                },
                indent=2,
            ),
            flush=True,
        )

    if stop_flags:
        (DERIVED / "STOP.json").write_text(json.dumps({"stop_flags": stop_flags}, indent=2) + "\n")
        raise SystemExit("STOP: " + "; ".join(stop_flags))

    # Write machine-readable outputs
    table_rows = []
    boot_rows = []
    for market, res in market_results.items():
        if res["stop"]:
            continue
        p = res["point"]
        ci = res["ci_bp"]
        boot = res["boot"]
        table_rows.append(
            {
                "Market": market,
                "Mean_C_Discovery_bp": p["mean_c_D_bp"],
                "Mean_C_Confirmation_bp": p["mean_c_H_bp"],
                "Transported_D_on_H_bp": p["C_H_from_D_bp"],
                "TE_D_to_H_bp": p["TE_D_to_H_bp"],
                "TE_CI_lo_bp": ci["TE_D_to_H"]["ci_lo_bp"],
                "TE_CI_hi_bp": ci["TE_D_to_H"]["ci_hi_bp"],
                "Std_D_bp": p["C_D_std_bp"],
                "Std_H_bp": p["C_H_std_bp"],
                "STD_DIFF_bp": p["STD_DIFF_bp"],
                "STD_DIFF_CI_lo_bp": ci["STD_DIFF"]["ci_lo_bp"],
                "STD_DIFF_CI_hi_bp": ci["STD_DIFF"]["ci_hi_bp"],
                "DELTA_pi_bp": p["DELTA_pi_bp"],
                "DELTA_pi_CI_lo_bp": ci["DELTA_pi"]["ci_lo_bp"],
                "DELTA_pi_CI_hi_bp": ci["DELTA_pi"]["ci_hi_bp"],
                "DELTA_p_bp": p["DELTA_p_bp"],
                "DELTA_p_CI_lo_bp": ci["DELTA_p"]["ci_lo_bp"],
                "DELTA_p_CI_hi_bp": ci["DELTA_p"]["ci_hi_bp"],
                "DELTA_m_bp": p["DELTA_m_bp"],
                "DELTA_m_CI_lo_bp": ci["DELTA_m"]["ci_lo_bp"],
                "DELTA_m_CI_hi_bp": ci["DELTA_m"]["ci_hi_bp"],
                "H0_ext_p_boot": boot["p_values"]["H0_ext"],
                "H0_int_p_boot": boot["p_values"]["H0_int"],
                "pattern": res["pattern"],
            }
        )
        for stat, pack in ci.items():
            boot_rows.append(
                {
                    "Market": market,
                    "statistic": stat,
                    "point_bp": p.get(f"{stat}_bp", p.get(stat)),
                    "ci_lo_bp": pack["ci_lo_bp"],
                    "ci_hi_bp": pack["ci_hi_bp"],
                    "n_valid": pack["n_valid"],
                    "invalid_share": pack["invalid_share"],
                    "BOOTSTRAP_SUPPORT_INSTABILITY": pack["BOOTSTRAP_SUPPORT_INSTABILITY"],
                }
            )
        for test in ("H0_ext", "H0_int"):
            boot_rows.append(
                {
                    "Market": market,
                    "statistic": test,
                    "point_bp": "",
                    "ci_lo_bp": "",
                    "ci_hi_bp": "",
                    "n_valid": "",
                    "invalid_share": boot["invalid_share"][test],
                    "BOOTSTRAP_SUPPORT_INSTABILITY": boot["invalid_share"][test] > INVALID_SHARE_GATE,
                    "p_boot": boot["p_values"][test],
                    "wald_T": boot["wald"][test]["T"] if boot["wald"][test] else "",
                    "wald_rank": boot["wald"][test]["rank"] if boot["wald"][test] else "",
                    "wald_condition": boot["wald"][test]["condition_kept"] if boot["wald"][test] else "",
                }
            )

    table_path = WORK / "RESPONSE_MAP_TRANSPORT_TABLE.csv"
    with table_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        w.writeheader()
        w.writerows(table_rows)

    boot_path = WORK / "RESPONSE_MAP_TRANSPORT_BOOTSTRAP_SUMMARY.csv"
    # Union of keys
    keys = []
    for row in boot_rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with boot_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(boot_rows)

    # JSON dump (maps as plain dicts)
    serializable = {}
    for market, res in market_results.items():
        serializable[market] = {
            "path_d": res["path_d"],
            "path_h": res["path_h"],
            "point": res["point"],
            "ci_bp": res["ci_bp"],
            "pattern": res["pattern"],
            "bin_rows": res["bin_rows"],
            "boot_invalid_share": res["boot"]["invalid_share"],
            "boot_p_values": res["boot"]["p_values"],
            "boot_wald": res["boot"]["wald"],
            "pi_bar": (0.5 * (res["maps_D"].pi + res["maps_H"].pi)).tolist(),
            "shapley_residual": res["point"]["shapley_residual"],
        }
    (DERIVED / "rmt_results.json").write_text(json.dumps(serializable, indent=2) + "\n")

    # Figure rendering is manuscript-presentation code, out of scope for
    # code (task section 20/25) -- skip rather than fail the run.
    try:
        make_figure(market_results, FIGURES / "figure_rmt_response_maps.png")
    except SystemExit as exc:
        print(f"(skipping figure rendering: {exc})")

    meta = {
        "seed": RMT_SEED,
        "reps": RMT_REPS,
        "invalid_share_gate": INVALID_SHARE_GATE,
        "markets": markets,
        "epistemic_status": "POST-CONFIRMATORY EXPLORATORY ANALYSIS",
        "beta2_untouched": True,
        "SIGN_REVERSED_untouched": True,
    }
    (DERIVED / "rmt_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("Wrote", table_path)
    print("Wrote", boot_path)
    print("Wrote figures under", FIGURES)


if __name__ == "__main__":
    main()
