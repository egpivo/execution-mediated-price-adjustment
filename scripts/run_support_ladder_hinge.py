#!/usr/bin/env python3
"""Driver for the hinge-side support-cap ladder (table_t6_support source).

REWRITE, unverified against the original -- see
src/research_core/methods/support_ladder_hinge.py docstring and
CODE_CORE_EXTRACTION_AUDIT.md section 6, item 1, before trusting output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core.methods.support_ladder_hinge import support_cap_ladder


def load_swap_sample(panel_path: Path):
    t = pq.read_table(
        panel_path,
        columns=["block_timestamp", "retained_hf", "has_focal_swap", "x", "correction_hf"],
    ).to_pandas()
    sub = t[t.retained_hf & t.has_focal_swap].copy()
    sub = sub[np.isfinite(sub.x) & np.isfinite(sub.correction_hf)]
    sub["day"] = (sub.block_timestamp.astype("int64") // 86400).astype("int64")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True, type=Path, help="canonical response panel parquet")
    ap.add_argument("--market", required=True)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    swap = load_swap_sample(args.panel)
    result = support_cap_ladder(
        swap.x.to_numpy(), swap.correction_hf.to_numpy(), swap.day.to_numpy()
    )
    result["market"] = args.market
    result["n_total"] = int(len(swap))
    args.out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
