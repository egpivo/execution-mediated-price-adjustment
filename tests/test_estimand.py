"""Unit tests for research_core.methods.estimand: the frozen 10-bin
support schedule (CODE_REVIEW.md repair #11) and core estimand formulas.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core.methods.estimand import (
    X_SUPPORT_BIN_FLAG_NAMES,
    build_x_support_flags,
    gamma_fee,
    hf_same_block_outcomes,
)

EXPECTED_BIN_NAMES = (
    "x_0_0_25", "x_0_25_0_5", "x_0_5_0_75", "x_0_75_1",
    "x_1_1_25", "x_1_25_1_5", "x_1_5_2", "x_2_3", "x_3_5", "x_gt_5",
)


def test_ten_bins_exactly():
    assert X_SUPPORT_BIN_FLAG_NAMES == EXPECTED_BIN_NAMES
    assert len(X_SUPPORT_BIN_FLAG_NAMES) == 10


@pytest.mark.parametrize(
    "x,expected_bin",
    [
        (0.0, "x_0_0_25"),
        (0.1, "x_0_0_25"),
        (0.25, "x_0_25_0_5"),      # half-open lower-inclusive: boundary goes to the UPPER bin
        (0.4999, "x_0_25_0_5"),
        (0.5, "x_0_5_0_75"),
        (0.75, "x_0_75_1"),
        (1.0, "x_1_1_25"),         # hinge breakpoint x=1 falls in [1,1.25)
        (1.25, "x_1_25_1_5"),
        (1.5, "x_1_5_2"),
        (2.0, "x_2_3"),
        (3.0, "x_3_5"),
        (5.0, "x_gt_5"),           # exactly the last finite edge -> open-ended tail bin
        (1e9, "x_gt_5"),
    ],
)
def test_boundary_values_fall_in_correct_bin(x, expected_bin):
    flags = build_x_support_flags(np.array([x]))
    hit = [name for name in EXPECTED_BIN_NAMES if flags[name][0]]
    assert hit == [expected_bin], f"x={x} matched {hit}, expected [{expected_bin}]"


def test_bins_are_mutually_exclusive_and_exhaustive_for_finite_nonneg_x():
    x = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 100.0])
    flags = build_x_support_flags(x)
    stacked = np.vstack([flags[n] for n in EXPECTED_BIN_NAMES])
    assert (stacked.sum(axis=0) == 1).all(), "every finite non-negative x must match exactly one bin"


def test_nan_inf_negative_rejected_no_silent_clipping():
    x = np.array([np.nan, np.inf, -np.inf, -1.0, -0.001])
    flags = build_x_support_flags(x)
    stacked = np.vstack([flags[n] for n in EXPECTED_BIN_NAMES])
    assert (stacked.sum(axis=0) == 0).all(), "NaN/inf/negative x must not match any bin"


def test_gamma_fee_matches_log_convention():
    assert gamma_fee("ETH5") == pytest.approx(-np.log(1 - 0.0005))
    assert gamma_fee("BASE1") == pytest.approx(-np.log(1 - 0.0001))


def test_hf_same_block_outcomes_c_zero_when_no_state_change():
    """Structural identity: if xA_pre == xA_post (no focal-pool swap moved
    the AMM state), correction_hf must be exactly 0, algebraically, not by
    a special-cased branch."""
    xstar = np.array([1.0, 1.0, 1.0])
    xA_pre = np.array([0.9, 0.9, 0.9])
    xA_post = xA_pre.copy()  # no swap: post state == pre state
    retained = np.array([True, True, True])
    out = hf_same_block_outcomes(xstar, xA_pre, xA_post, retained, "ETH5")
    assert np.allclose(out["correction_hf_same_block"], 0.0)


def test_hf_same_block_outcomes_x_is_nonnegative_and_unretained_is_nan():
    xstar = np.array([1.0, 1.0])
    xA_pre = np.array([0.5, 0.5])
    xA_post = np.array([0.6, 0.6])
    retained = np.array([True, False])
    out = hf_same_block_outcomes(xstar, xA_pre, xA_post, retained, "ETH5")
    assert out["x"][0] >= 0
    assert np.isnan(out["x"][1])
