"""Bounded unit tests for Response-Map Transport core identities."""

import numpy as np
import pytest

from rmt_core import (
    BIN_EDGES,
    N_BINS,
    F,
    aggregate_day_stats,
    assign_bins,
    maps_from_counts,
    require_finite_maps,
    require_finite_residual,
    shapley_decomposition,
    standardized_difference,
    transport_error,
)


def _maps(pi, p, m):
    # Construct BinMaps via counts that imply these (pi,p,m).
    # Choose total N=1000 for convenience.
    n = np.round(np.asarray(pi, dtype=np.float64) * 1000.0)
    # Fix rounding so sum(n)=1000
    n[-1] = 1000 - n[:-1].sum()
    n = np.maximum(n, 1.0)  # ensure p defined
    # renormalize intent: rebuild pi from n
    p = np.asarray(p, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    n_swap = p * n
    sum_c_swap = m * n_swap
    return maps_from_counts(n, n_swap, sum_c_swap)


def test_pi_sums_to_one():
    rng = np.random.default_rng(0)
    n = rng.integers(10, 100, size=N_BINS).astype(float)
    n_swap = np.minimum(n, rng.integers(1, 50, size=N_BINS).astype(float))
    sum_c = rng.normal(size=N_BINS) * n_swap
    maps = maps_from_counts(n, n_swap, sum_c)
    assert abs(maps.pi.sum() - 1.0) < 1e-12


def test_pooled_support_weights_sum_to_one():
    pi_d = np.full(N_BINS, 1.0 / N_BINS)
    pi_h = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])
    p = np.linspace(0.1, 0.9, N_BINS)
    m = np.linspace(-1e-4, 5e-4, N_BINS)
    maps_d = _maps(pi_d, p, m)
    maps_h = _maps(pi_h, p, m)
    std = standardized_difference(maps_d, maps_h)
    assert abs(std["pi_bar_sum"] - 1.0) < 1e-12


def test_transport_identity_mean_c():
    rng = np.random.default_rng(1)
    n = rng.integers(50, 200, size=N_BINS).astype(float)
    n_swap = np.clip(rng.integers(10, 150, size=N_BINS).astype(float), 1, None)
    n_swap = np.minimum(n_swap, n)
    sum_c = rng.normal(size=N_BINS) * 1e-4 * n_swap
    maps = maps_from_counts(n, n_swap, sum_c)
    assert abs(maps.mean_c - maps.mean_c_from_bins) < 1e-12
    assert abs(maps.mean_c - F(maps.pi, maps.p, maps.m)) < 1e-12


def test_shapley_exact_adding_up():
    pi_d = np.full(N_BINS, 1.0 / N_BINS)
    pi_h = np.array([0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
    p_d = np.linspace(0.2, 0.8, N_BINS)
    p_h = np.linspace(0.3, 0.9, N_BINS)
    m_d = np.linspace(-2e-4, 4e-4, N_BINS)
    m_h = np.linspace(-1e-4, 3e-4, N_BINS)
    maps_d = _maps(pi_d, p_d, m_d)
    maps_h = _maps(pi_h, p_h, m_h)
    sh = shapley_decomposition(maps_d, maps_h)
    assert abs(sh["shapley_residual"]) < 1e-12
    assert abs(sh["DELTA"] - (sh["DELTA_pi"] + sh["DELTA_p"] + sh["DELTA_m"])) < 1e-12


def test_synthetic_no_change_all_zero():
    pi = np.full(N_BINS, 1.0 / N_BINS)
    p = np.linspace(0.2, 0.9, N_BINS)
    m = np.linspace(-1e-4, 5e-4, N_BINS)
    maps = _maps(pi, p, m)
    te = transport_error(maps.pi, maps, maps)
    std = standardized_difference(maps, maps)
    sh = shapley_decomposition(maps, maps)
    assert abs(te["TE_D_to_H"]) < 1e-15
    assert abs(std["STD_DIFF"]) < 1e-15
    assert abs(sh["DELTA"]) < 1e-15
    assert abs(sh["DELTA_pi"]) < 1e-15
    assert abs(sh["DELTA_p"]) < 1e-15
    assert abs(sh["DELTA_m"]) < 1e-15


def test_synthetic_one_factor_only_pi():
    pi_d = np.full(N_BINS, 1.0 / N_BINS)
    pi_h = np.array([0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    p = np.linspace(0.2, 0.8, N_BINS)
    m = np.linspace(1e-4, 5e-4, N_BINS)
    maps_d = _maps(pi_d, p, m)
    maps_h = _maps(pi_h, p, m)  # p,m identical intent
    # Force exact same p,m on the realized maps (rounding in _maps can perturb)
    maps_h = type(maps_h)(
        n=maps_h.n,
        n_swap=maps_h.n_swap,
        pi=maps_h.pi,
        p=maps_d.p.copy(),
        m=maps_d.m.copy(),
        r=maps_d.p * maps_d.m,
        mean_c=F(maps_h.pi, maps_d.p, maps_d.m),
        mean_c_from_bins=F(maps_h.pi, maps_d.p, maps_d.m),
        p_defined=maps_d.p_defined,
        m_defined=maps_d.m_defined,
    )
    sh = shapley_decomposition(maps_d, maps_h)
    assert abs(sh["DELTA_p"]) < 1e-12
    assert abs(sh["DELTA_m"]) < 1e-12
    assert abs(sh["DELTA_pi"] - sh["DELTA"]) < 1e-12


def test_synthetic_one_factor_only_p():
    pi = np.full(N_BINS, 1.0 / N_BINS)
    p_d = np.linspace(0.2, 0.5, N_BINS)
    p_h = np.linspace(0.4, 0.9, N_BINS)
    m = np.linspace(1e-4, 4e-4, N_BINS)
    maps_d = _maps(pi, p_d, m)
    maps_h = _maps(pi, p_h, m)
    maps_h = type(maps_h)(
        n=maps_h.n,
        n_swap=maps_h.n_swap,
        pi=maps_d.pi.copy(),
        p=maps_h.p,
        m=maps_d.m.copy(),
        r=maps_h.p * maps_d.m,
        mean_c=F(maps_d.pi, maps_h.p, maps_d.m),
        mean_c_from_bins=F(maps_d.pi, maps_h.p, maps_d.m),
        p_defined=maps_h.p_defined,
        m_defined=maps_d.m_defined,
    )
    sh = shapley_decomposition(maps_d, maps_h)
    assert abs(sh["DELTA_pi"]) < 1e-12
    assert abs(sh["DELTA_m"]) < 1e-12
    assert abs(sh["DELTA_p"] - sh["DELTA"]) < 1e-12


def test_synthetic_one_factor_only_m():
    pi = np.full(N_BINS, 1.0 / N_BINS)
    p = np.linspace(0.2, 0.8, N_BINS)
    m_d = np.linspace(1e-4, 2e-4, N_BINS)
    m_h = np.linspace(3e-4, 6e-4, N_BINS)
    maps_d = _maps(pi, p, m_d)
    maps_h = _maps(pi, p, m_h)
    maps_h = type(maps_h)(
        n=maps_h.n,
        n_swap=maps_h.n_swap,
        pi=maps_d.pi.copy(),
        p=maps_d.p.copy(),
        m=maps_h.m,
        r=maps_d.p * maps_h.m,
        mean_c=F(maps_d.pi, maps_d.p, maps_h.m),
        mean_c_from_bins=F(maps_d.pi, maps_d.p, maps_h.m),
        p_defined=maps_d.p_defined,
        m_defined=maps_h.m_defined,
    )
    sh = shapley_decomposition(maps_d, maps_h)
    assert abs(sh["DELTA_pi"]) < 1e-12
    assert abs(sh["DELTA_p"]) < 1e-12
    assert abs(sh["DELTA_m"] - sh["DELTA"]) < 1e-12


def test_assign_bins_edges():
    x = np.array([0.0, 0.249999, 0.25, 0.999, 1.0, 5.0, 100.0])
    idx = assign_bins(x)
    assert idx.tolist() == [0, 0, 1, 3, 4, 9, 9]
    assert BIN_EDGES.shape[0] == N_BINS + 1


def test_aggregate_day_stats_matches_direct():
    # two days, two bins populated
    n = np.array([[10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0.0, 20, 0, 0, 0, 0, 0, 0, 0, 0]])
    n_swap = np.array([[4.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0.0, 10, 0, 0, 0, 0, 0, 0, 0, 0]])
    sum_c = np.array([[0.001, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0.0, 0.004, 0, 0, 0, 0, 0, 0, 0, 0]])
    # fill remaining bins with tiny mass so all p defined? Actually other bins empty.
    # For this test only check F on populated bins via mean_c.
    maps = aggregate_day_stats(n, n_swap, sum_c, day_counts=np.array([1.0, 2.0]))
    # day0*1 + day1*2 => n0=10, n1=40, total=50
    assert maps.n[0] == 10
    assert maps.n[1] == 40
    assert maps.n_swap[0] == 4
    assert maps.n_swap[1] == 20
    assert abs(maps.mean_c - (0.001 + 2 * 0.004) / 50.0) < 1e-15


# --- CODE_REVIEW.md repair #9: fail-closed on non-finite maps/residuals ---

def _maps_with_zero_swap_bin():
    """n>0 in every bin but n_swap==0 in bin 0 -- the exact zero-swap-bin
    case the reviewer identified as untested. m is undefined (NaN) there."""
    n = np.full(N_BINS, 100.0)
    n_swap = np.full(N_BINS, 40.0)
    n_swap[0] = 0.0  # bin 0: no swaps observed
    sum_c_swap = n_swap * 1e-4
    return maps_from_counts(n, n_swap, sum_c_swap)


def test_zero_swap_bin_leaves_m_undefined_not_zero():
    maps = _maps_with_zero_swap_bin()
    assert np.isnan(maps.m[0]), "m for a zero-swap bin must be NaN, not silently zero-filled"
    assert maps.p[0] == 0.0, "p is well-defined (0) even when m is not"


def test_require_finite_maps_raises_on_zero_swap_bin():
    maps_ok = _maps(np.full(10, 0.1), np.full(10, 0.5), np.linspace(-1, 8, 10) * 1e-4)
    maps_bad = _maps_with_zero_swap_bin()
    with pytest.raises(ValueError, match="non-finite"):
        require_finite_maps(maps_ok, maps_bad)
    with pytest.raises(ValueError, match="non-finite"):
        require_finite_maps(maps_bad, maps_ok)


def test_require_finite_maps_passes_when_all_finite():
    maps_ok = _maps(np.full(10, 0.1), np.full(10, 0.5), np.linspace(-1, 8, 10) * 1e-4)
    require_finite_maps(maps_ok, maps_ok)  # must not raise


def test_shapley_on_undefined_map_produces_nan_residual_not_silent_number():
    """Demonstrates the actual failure mode: calling shapley_decomposition
    directly on an undefined map (skipping the guard) produces a NaN
    residual -- proving the guard is necessary, not decorative."""
    maps_ok = _maps(np.full(10, 0.1), np.full(10, 0.5), np.linspace(-1, 8, 10) * 1e-4)
    maps_bad = _maps_with_zero_swap_bin()
    sh = shapley_decomposition(maps_ok, maps_bad)
    assert np.isnan(sh["shapley_residual"])
    assert np.isnan(sh["DELTA"])


def test_require_finite_residual_raises_on_nan():
    with pytest.raises(ValueError, match="non-finite residual"):
        require_finite_residual(float("nan"), tol=1e-12)


def test_require_finite_residual_raises_on_out_of_tolerance():
    with pytest.raises(ValueError, match="exceeds tolerance"):
        require_finite_residual(1.0, tol=1e-12)


def test_require_finite_residual_passes_within_tolerance():
    require_finite_residual(1e-15, tol=1e-12)  # must not raise
