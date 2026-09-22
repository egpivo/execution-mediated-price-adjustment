"""Unit tests for eth_pool_events.py's fail-closed RPC ingestion behavior
(CODE_REVIEW.md repair #1) and the block-hash consistency check.

No network access -- rpc() and block_timestamp() are monkeypatched.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import research_core.ingestion.eth_pool_events as mod  # noqa: E402


def _fake_log(block_number: int, block_hash: str = "0xaaa") -> dict:
    return {
        "address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        "topics": [mod.TOPICS["Initialize"]],
        "data": "0x" + "00" * 32 + "00" * 32,
        "blockNumber": hex(block_number),
        "blockHash": block_hash,
        "transactionHash": "0xtx",
        "transactionIndex": "0x0",
        "logIndex": "0x0",
    }


def test_check_block_hash_consistency_passes_on_consistent_rows():
    rows = [
        {"block_number": 100, "block_hash": "0xabc"},
        {"block_number": 100, "block_hash": "0xabc"},
        {"block_number": 101, "block_hash": "0xdef"},
    ]
    mod.check_block_hash_consistency(rows)  # must not raise


def test_check_block_hash_consistency_raises_on_mismatch():
    rows = [
        {"block_number": 100, "block_hash": "0xabc"},
        {"block_number": 100, "block_hash": "0xDIFFERENT"},
    ]
    with pytest.raises(RuntimeError, match="block_hash mismatch"):
        mod.check_block_hash_consistency(rows)


def test_pull_range_recovers_from_transient_failure(monkeypatch, tmp_path):
    """A transient failure followed by success on retry must NOT advance
    `cur` past the range and must NOT drop any rows -- the range is
    eventually served, not skipped."""
    calls = {"n": 0}

    def fake_rpc(method, params, retries=None):
        if method == "eth_getBlockByNumber":
            return {"timestamp": hex(1_700_000_000)}
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated transient RPC failure")
        return [_fake_log(100)]

    monkeypatch.setattr(mod, "rpc", fake_rpc)
    monkeypatch.setattr(mod, "write_part", lambda rows, run, chain: None)

    result = mod.pull_range(
        pools=["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"],
        event_names=["Initialize"],
        frm=100, tip=100, step=1000,
        chain="ethereum", chain_id=1,
    )
    assert result["done"] is True
    assert result["new_rows"] == 1  # the retried call's row was captured, not skipped


def test_pull_range_fails_closed_after_exhausting_step_halving(monkeypatch, tmp_path):
    """A range that keeps failing even once step has been halved to the
    50-block minimum must raise, not silently advance past it, and must
    record a gap manifest entry (CODE_REVIEW.md repair #1)."""
    def always_fails(method, params, retries=None):
        if method == "eth_getBlockByNumber":
            return {"timestamp": hex(1_700_000_000)}
        raise RuntimeError("simulated persistent RPC failure")

    monkeypatch.setattr(mod, "rpc", always_fails)
    monkeypatch.setattr(mod, "write_part", lambda rows, run, chain: None)
    monkeypatch.setattr(mod, "pool_events_partition_dir", lambda chain: tmp_path / "pool_events")

    with pytest.raises(RuntimeError, match="after exhausting retries"):
        mod.pull_range(
            pools=["0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"],
            event_names=["Initialize"],
            frm=100, tip=200, step=1000,  # will halve down to 50 then still fail
            chain="ethereum", chain_id=1,
        )

    gap_path = tmp_path / "manifests" / "ingestion_gaps.jsonl"
    assert gap_path.is_file(), "a failed range must be recorded in the gap manifest before raising"
    gap = json.loads(gap_path.read_text().splitlines()[0])
    assert gap["chain"] == "ethereum"
    assert gap["from_block"] == 100
