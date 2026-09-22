"""Frozen numerical regression tests (task requirement: reproduce the paper's
canonical numerical inputs to floating-point tolerance, no network calls).

Two tiers:

1. Fixture-only checks (always run): the frozen result JSON/CSV checked into
   tests/fixtures/ must contain exactly the headline numbers from the paper.
   This does not re-run any estimation -- it only pins the known-good values
   so a future accidental edit to the fixtures is caught.

2. Live end-to-end checks (skipped if the external response-panel cache is
   not present on this machine): re-run scripts/run_confirmation.py's
   run_market() against the real cached panels and assert the freshly
   computed beta2/CI matches the frozen fixture. This is the actual
   "reproduce the frozen value from canonical inputs" test. The panel cache
   lives outside git (per CODE_CORE_EXTRACTION_AUDIT.md data-rights /
   size policy) at the path below; if it is absent the test is SKIPPED, not
   failed, and does not fabricate a pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from research_core.paths import confirmatory_panels_dir  # noqa: E402

# Cache produced by the original pipeline; not part of this repo (raw/derived
# data policy, see README "Data rights"). Resolved via research_core.paths
# (set RESEARCH_CORE_EXTERNAL_CACHE); will be absent on a fresh clone.
PANEL_CACHE = confirmatory_panels_dir()

FROZEN_BETA2 = {
    "ETH5": (-3.653, -4.146, -3.026),
    "BASE5": (-1.478, -1.775, -1.107),
    "BASE1": (-1.680, -1.896, -1.407),
}

FROZEN_RMT = {
    "ETH5": dict(TE=-2.21198, STD_DIFF=-1.14083, DELTA_pi=1.25606, DELTA_p=0.01113, DELTA_m=-1.17089),
    "BASE5": dict(TE=-1.04663, STD_DIFF=-0.52453, DELTA_pi=0.48914, DELTA_p=-0.08253, DELTA_m=-0.37025),
}

TOL_BP = 0.001  # matches the 3-decimal precision the values are quoted at


def test_fixture_confirmation_matches_frozen_brief():
    data = json.loads((FIXTURES / "frozen_confirmation_results.json").read_text())
    for market, (beta2, lo, hi) in FROZEN_BETA2.items():
        row = data[market]
        assert row["beta2"] * 1e4 == pytest.approx(beta2, abs=TOL_BP)
        assert row["beta2_ci95"][0] * 1e4 == pytest.approx(lo, abs=TOL_BP)
        assert row["beta2_ci95"][1] * 1e4 == pytest.approx(hi, abs=TOL_BP)


def test_fixture_rmt_matches_frozen_brief():
    data = json.loads((FIXTURES / "frozen_rmt_results.json").read_text())
    for market, expect in FROZEN_RMT.items():
        point = data[market]["point"]
        assert point["TE_D_to_H_bp"] == pytest.approx(expect["TE"], abs=TOL_BP)
        assert point["STD_DIFF_bp"] == pytest.approx(expect["STD_DIFF"], abs=TOL_BP)
        assert point["DELTA_pi_bp"] == pytest.approx(expect["DELTA_pi"], abs=TOL_BP)
        assert point["DELTA_p_bp"] == pytest.approx(expect["DELTA_p"], abs=TOL_BP)
        assert point["DELTA_m_bp"] == pytest.approx(expect["DELTA_m"], abs=TOL_BP)


@pytest.mark.skipif(not PANEL_CACHE.is_dir(), reason="external response-panel cache not present on this machine")
def test_live_support_ladder_hinge_matches_recovered_frozen_json():
    """The REWRITE (support_ladder_hinge.py) against a frozen copy of the
    ORIGINAL script's own output (section_e_raw.json), recovered after the
    rewrite was written -- see that module's docstring."""
    import pyarrow.parquet as pq

    from research_core.methods.support_ladder_hinge import support_cap_ladder

    frozen = json.loads((FIXTURES / "frozen_section_e_hinge_support.json").read_text())["ETH5"]["confirmation"]["support_caps"]

    t = pq.read_table(
        PANEL_CACHE / "eth5_pre_response_panel.parquet",
        columns=["block_timestamp", "retained_hf", "has_focal_swap", "x", "correction_hf"],
    ).to_pandas()
    sub = t[t.retained_hf & t.has_focal_swap]
    sub = sub[(sub.x.notna()) & (sub.correction_hf.notna())]
    day = (sub.block_timestamp.astype("int64") // 86400).astype("int64")

    result = support_cap_ladder(
        sub.x.to_numpy(), sub.correction_hf.to_numpy(), day.to_numpy(), reps=500, seed=20260829
    )["support_caps"]

    for label, expect in frozen.items():
        if "beta2" not in expect:
            continue
        got = result[label]
        assert got["n"] == expect["n"]
        assert got["beta2"] == pytest.approx(expect["beta2"], abs=1e-12)
        assert got["beta2_ci95"][0] == pytest.approx(expect["beta2_ci95"][0], abs=1e-12)
        assert got["beta2_ci95"][1] == pytest.approx(expect["beta2_ci95"][1], abs=1e-12)


@pytest.mark.skipif(not PANEL_CACHE.is_dir(), reason="external response-panel cache not present on this machine")
def test_live_confirmation_reproduces_frozen_beta2():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_confirmation", Path(__file__).resolve().parents[2] / "scripts" / "run_confirmation.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    files = {
        "ETH5": "eth5_pre_response_panel.parquet",
        "BASE5": "base5_pre_response_panel.parquet",
        "BASE1": "base1_pre_response_panel.parquet",
    }
    for market, (beta2, lo, hi) in FROZEN_BETA2.items():
        result = mod.run_market(PANEL_CACHE / files[market], 86400, market)
        assert result["beta2"] * 1e4 == pytest.approx(beta2, abs=TOL_BP)
        assert result["beta2_ci95"][0] * 1e4 == pytest.approx(lo, abs=TOL_BP)
        assert result["beta2_ci95"][1] * 1e4 == pytest.approx(hi, abs=TOL_BP)
