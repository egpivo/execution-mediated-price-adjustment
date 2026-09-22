"""Minimal coverage-manifest helpers for the raw pool-events hive.

Provenance: a small vendored subset of `amm_data.coverage`
(`/Users/joseph/amm-data/src/amm_data/coverage.py::load_coverage,
load_universe, WATERMARK_EVENT`), copied rather than imported so
research-core has no runtime dependency on the separate `amm-data` package
(CODE_REVIEW.md repair #2). Only the two pure JSON-reading functions
`eth_pool_events.py` actually calls are reproduced here -- the full
`amm_data.coverage` module also builds/verifies the manifest from a DuckDB
scan of the lake, which is out of scope for this paper's pipeline and is
not vendored.
"""
from __future__ import annotations

import json
from pathlib import Path

from research_core.paths import coverage_manifest_path

WATERMARK_EVENT = "Swap"


def load_coverage(path: Path | None = None, chain: str = "ethereum") -> dict:
    p = path or coverage_manifest_path(chain=chain)
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def load_universe(path: Path | None) -> set[str] | None:
    if path is None or not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return {str(p).lower() for p in raw}
    if isinstance(raw, dict):
        addrs: set[str] = set()
        for key in ("pools", "treated_matched", "controls", "unmatched_treated", "crossvenue_forks"):
            for p in raw.get(key) or []:
                addrs.add(str(p).lower())
        if addrs:
            return addrs
    return None
