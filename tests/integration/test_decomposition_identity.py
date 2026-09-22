"""Wraps scripts/qa_decomposition_identity.py (the original phase_h_qa.py,
kept verbatim except for its output path -- see that file's header comment
in CODE_CORE_EXTRACTION_PLAN.md KEEP table).

Verifies E[C|x] = P(S=1|x)*E[C|S=1,x] holds to numerical precision on the
real cached confirmation-window panels for ETH5/BASE5/BASE1. Skips if the
external panel cache is not present (see README "Data rights").
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from research_core.paths import confirmatory_panels_dir  # noqa: E402

PANEL_CACHE = confirmatory_panels_dir()


@pytest.mark.skipif(not PANEL_CACHE.is_dir(), reason="external response-panel cache not present on this machine")
def test_decomposition_identity_holds_on_real_panels():
    script = Path(__file__).resolve().parents[2] / "scripts" / "qa_decomposition_identity.py"
    spec = importlib.util.spec_from_file_location("qa_decomposition_identity", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # runs the script; raises on failure

    out = Path(__file__).resolve().parents[2] / "outputs" / "tables" / "phase_h_qa_results.json"
    results = json.loads(out.read_text())
    for market, r in results.items():
        assert r["pool_address_correct"] is True
        assert r["state_continuity_violations"] == 0
        assert r["K_implies_S_violations"] == 0
        assert r["no_swap_max_abs_correction"] == 0.0
        assert abs(r["decomposition_max_residual"]) < 1e-9
