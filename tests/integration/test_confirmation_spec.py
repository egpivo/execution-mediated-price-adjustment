#!/usr/bin/env python3
"""Validates that the frozen spec (CONFIRMATORY_SPEC.json) was actually
honored, AND that the Confirmation/Discovery window boundary holds against
real panel timestamps, not just against a manifest asserting its own dates.

CODE_REVIEW.md repair #6: the previous version of this file defined only a
`main()` (0 pytest items collected -- silently never ran under `pytest`),
and its window-exclusion check read a panel's `block_number` column into a
variable it never used, then only compared one JSON manifest's declared
dates against a hardcoded dict of the same dates -- i.e. it checked a
manifest agrees with itself, not that any real data respects the boundary.
Both defects are fixed below: real `def test_*` functions, and
`test_confirmation_strictly_precedes_discovery` loads actual
`block_timestamp` columns from both windows' cached panels and asserts
`max(confirmation ts) < min(discovery ts)`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from research_core.paths import confirmatory_panels_dir, hf_response_full_derived_dir  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
DISCOVERY_DATES = {"start": "2026-07-06", "end": "2026-08-05"}
CONFIRMATION_PANELS = confirmatory_panels_dir()
DISCOVERY_PANEL = hf_response_full_derived_dir() / "eth5_response_panel_hf_v1.parquet"


def _spec_and_results():
    spec = json.loads((FIXTURES / "CONFIRMATORY_SPEC.json").read_text())
    results = json.loads((FIXTURES / "LONG_CONFIRMATORY_RESULTS.json").read_text())
    return spec, results


def test_spec_declares_fixed_breakpoint_no_search():
    spec, _ = _spec_and_results()
    assert spec["primary_model"]["breakpoint_fixed_at"] == 1.0
    assert spec["primary_model"]["no_search"] is True


def test_results_bootstrap_reps_seed_match_spec():
    spec, results = _spec_and_results()
    for market, r in results.items():
        assert r["bootstrap_reps"] == spec["inference"]["reps"], market
        assert r["bootstrap_seed"] == spec["inference"]["seed"], market


def test_results_preregistered_sign_matches_spec():
    spec, results = _spec_and_results()
    for market, r in results.items():
        assert r["preregistered_sign_b2"] == spec["primary_model"]["preregistered_sign_b2"], market


def test_result_markets_match_spec_declaration():
    spec, results = _spec_and_results()
    expected = set(spec["primary_model"]["markets"]) | {"BASE1"}
    assert set(results.keys()) == expected


def test_long_panel_qa_gate_passed():
    qa = json.loads((FIXTURES / "LONG_PANEL_QA.json").read_text())
    assert qa["gate"] == "LONG_PANEL_QA_PASS"


def test_discovery_manifest_dates_match_expected():
    manifest = json.loads((FIXTURES / "DISCOVERY_SAMPLE_MANIFEST.json").read_text())
    assert manifest["discovery_window"] == DISCOVERY_DATES


@pytest.mark.skipif(
    not (CONFIRMATION_PANELS / "eth5_pre_panel.parquet").is_file() or not DISCOVERY_PANEL.is_file(),
    reason="external confirmation/discovery panel cache not present on this machine",
)
def test_confirmation_strictly_precedes_discovery():
    """The real, data-level check the reviewer asked for: not a manifest
    comparing itself, but max(Confirmation block_timestamp) actually less
    than min(Discovery block_timestamp) in the cached panels."""
    # Confirmation panel carries synthetic/interpolated timestamps for blocks
    # without a real header (has_real_timestamp=False); only real timestamps
    # are meaningful for a window-boundary comparison.
    conf = pq.read_table(
        CONFIRMATION_PANELS / "eth5_pre_panel.parquet",
        columns=["block_timestamp", "has_real_timestamp"],
    ).to_pandas()
    confirmation_ts = conf.loc[conf["has_real_timestamp"], "block_timestamp"].tolist()
    discovery_ts = pq.read_table(
        DISCOVERY_PANEL, columns=["block_timestamp"]
    ).column("block_timestamp").to_pylist()

    assert confirmation_ts, "confirmation panel has no rows with a real timestamp"
    assert discovery_ts, "discovery panel has no rows"
    assert max(confirmation_ts) < min(discovery_ts), (
        f"windows overlap: max(confirmation)={max(confirmation_ts)} "
        f">= min(discovery)={min(discovery_ts)}"
    )
