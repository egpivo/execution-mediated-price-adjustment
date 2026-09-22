#!/usr/bin/env python3
"""Merge confirmation + discovery block headers for Tier A overlap."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from research_core.paths import (
    confirmatory_panels_dir,
    public_reference_audit_block_headers_dir,
    service_primitives_block_headers_dir,
)


def _paths() -> dict[str, dict[str, Path]]:
    return {
        "ethereum": {
            "pre": confirmatory_panels_dir() / "eth_pre_block_headers.csv",
            "disc": public_reference_audit_block_headers_dir() / "window_20260706_20260805.csv",
        },
        "base": {
            "pre": confirmatory_panels_dir() / "base_pre_block_headers.csv",
            "disc": service_primitives_block_headers_dir() / "base_window_20260706_20260805.csv",
        },
    }


def merge_chain(chain: str) -> pd.DataFrame:
    paths = _paths()[chain]
    pre = pd.read_csv(paths["pre"])
    disc = pd.read_csv(paths["disc"])
    pre = pre.rename(columns={"timestamp_unix": "block_timestamp"})
    if "block_timestamp" not in disc.columns:
        disc = disc.rename(columns={"timestamp_unix": "block_timestamp"})
    keep = ["block_number", "block_timestamp"]
    pre = pre[keep]
    disc = disc[keep]
    out = pd.concat([pre, disc], ignore_index=True).drop_duplicates("block_number").sort_values("block_number")
    if out.block_number.diff().dropna().lt(1).any():
        raise ValueError(f"non-monotonic block numbers for {chain}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for chain in ("ethereum", "base"):
        df = merge_chain(chain)
        out = args.output_dir / f"{chain}_tier_a_block_headers.csv"
        df.to_csv(out, index=False)
        print(f"{chain}: {len(df):,} blocks, {df.block_number.min()}..{df.block_number.max()} -> {out}")


if __name__ == "__main__":
    main()
