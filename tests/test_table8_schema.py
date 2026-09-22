"""Regression tests for run_rmt.py's parse_table8/verify_table8 against the
manuscript's full bin decomposition table schema (N, N_S, P(S|x), E[C|S,x],
E[C|x], Product, Residual). Guards against the exact drift class this test
was added to catch: a manuscript/generator column-schema change silently
breaking the verification parser (round-2 cold-review closure added an N_S
column; this parser previously assumed the pre-N_S 6-value-column layout)."""

from pathlib import Path

import pytest

import run_rmt_analysis as run_rmt

MANUSCRIPT_TABLE8 = run_rmt.MANUSCRIPT_TABLE8


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "table.tex"
    p.write_text(text)
    return p


GOOD_ROW = (
    r"ETH5 Discovery & [0,0.25) & 38,432 & 10,359 & 0.270 & -0.226 & -0.061 & -0.061 & 0.00e+00 \\"
    "\n"
)


def test_parses_current_manuscript_table_without_error():
    """The actual committed manuscript table must parse cleanly."""
    if not MANUSCRIPT_TABLE8.exists():
        pytest.skip("manuscript/generated/table_a_decomposition_bins.tex not present")
    table8 = run_rmt.parse_table8(MANUSCRIPT_TABLE8)
    assert len(table8) == 40  # 2 primary markets x 2 windows x 10 bins
    for market in ("ETH5", "BASE5"):
        for window in ("discovery", "confirmation"):
            for lab in run_rmt.BIN_LABELS:
                assert (market, window, lab) in table8


def test_parses_n_and_n_swap_as_exact_integers_not_rounded_from_p(tmp_path):
    path = _write(tmp_path, GOOD_ROW)
    table8 = run_rmt.parse_table8(path)
    row = table8[("ETH5", "discovery", "[0,0.25)")]
    assert row["n"] == 38432
    assert row["n_swap"] == 10359
    # N_S must come from the stored integer column, not round(N * displayed p).
    assert row["n_swap"] != round(row["n"] * row["p"])


def test_fails_closed_on_missing_column(tmp_path):
    """Dropping the N_S column (simulating a generator/parser schema
    mismatch) must raise, not silently misparse the remaining columns."""
    bad = GOOD_ROW.replace("10,359 & ", "")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="schema drift"):
        run_rmt.parse_table8(path)


def test_fails_closed_on_extra_column(tmp_path):
    bad = GOOD_ROW.replace("0.00e+00", "0.00e+00 & 999")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="schema drift"):
        run_rmt.parse_table8(path)


def test_fails_closed_on_non_integer_n(tmp_path):
    bad = GOOD_ROW.replace("38,432", "not_a_number")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="integer-parse failure"):
        run_rmt.parse_table8(path)


def test_fails_closed_when_n_swap_exceeds_n(tmp_path):
    bad = GOOD_ROW.replace("38,432 & 10,359", "38,432 & 99,999")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="N_S > N"):
        run_rmt.parse_table8(path)


def test_fails_closed_when_displayed_p_disagrees_with_n_swap_over_n(tmp_path):
    # N_S/N = 10359/38432 = 0.2695...; displaying p=0.900 disagrees well
    # beyond the 3-decimal display-rounding tolerance.
    bad = GOOD_ROW.replace("0.270", "0.900")
    path = _write(tmp_path, bad)
    with pytest.raises(ValueError, match="disagrees with displayed"):
        run_rmt.parse_table8(path)


def test_verify_table8_checks_n_swap_exactly():
    """verify_table8 must compare maps.n_swap against the table's n_swap
    exactly (TABLE8_TOL_N == 0), not silently skip it."""
    class FakeMaps:
        n = [38432]
        n_swap = [10359]
        p = [0.2695410074937552]
        m = [-0.22576135018087184e-4]
        r = [-0.22576135018087184e-4 * 0.2695410074937552]

    table8 = {
        ("ETH5", "discovery", run_rmt.BIN_LABELS[0]): {
            "n": 38432,
            "n_swap": 10358,  # deliberately off by one
            "p": 0.2695410074937552,
            "m_bp": -0.226,
            "product_bp": -0.061,
        }
    }
    issues = run_rmt.verify_table8("ETH5", "discovery", FakeMaps(), table8)
    assert any("n_swap" in issue for issue in issues)
