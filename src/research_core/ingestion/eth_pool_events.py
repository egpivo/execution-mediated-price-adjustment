"""Generic Uniswap v3 pool-event pull into universal hive parquet (append-only).

Writes new part-* files under the chain hive layout, e.g.:
  ethereum: $AMM_UNIVERSAL_ROOT/raw/pool_events/event=.../year=.../month=.../
  base:     $AMM_UNIVERSAL_ROOT/base/raw/pool_events/chain_id=8453/event=.../...
"""

import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from research_core.coverage import WATERMARK_EVENT, load_coverage, load_universe
from research_core.paths import (
    bootstrap_env,
    chain_config,
    ensure_universal_dirs,
    normalize_chain,
    pool_events_partition_dir,
)

TOPICS = {
    "Swap": "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67",
    "Mint": "0x7a53080ba414158be7ec69b987b5fb7d07dee101fe85488f0853ae16239d0bde",
    "Burn": "0x0c396cd989a39f4459b5fa1aed6a9a8dcdbc45908acfd67e028cd568da98982c",
    "Collect": "0x70935338e69775456a85ddef226c395fb668b63fa0115f5f20610b388e6ca9c0",
    "Initialize": "0x98636036cb66a9c19a37435efc1e9014219207740de06c78508615fcc55cf675",
}
TOPIC_TO_EVENT = {v.lower(): k for k, v in TOPICS.items()}

ENVELOPE_COLS = [
    "chain_id",
    "pool_address",
    "event_name",
    "block_number",
    "block_hash",
    "block_timestamp",
    "transaction_hash",
    "transaction_index",
    "log_index",
    "decoded_json",
    "raw_topics_json",
    "raw_data",
    "extraction_batch",
    "extraction_run",
    "timestamp_source",
]

# Per-chain pull context (set at start of run()).
_CHAIN = "ethereum"
_CHAIN_ID = 1
_RPC: str | None = None
_RPC_SLEEP = 0.0

# Default Collect bootstrap lower bounds (Uniswap v3 factory deployment era).
DEFAULT_COLLECT_FROM = {"ethereum": 12_369_621, "base": 1_371_680}


def rpcurl(chain: str | None = None) -> str:
    """Resolve the archival RPC URL from the environment (.env or exported).

    No hardcoded personal/host fallback path (CODE_REVIEW.md repair #5) --
    the only source of truth is ALCHEMY_ETHEREUM_URL / ALCHEMY_BASE_URL,
    with Base allowed to derive from the Ethereum URL by hostname
    substitution when only the Ethereum one is set (same provider, common
    setup, not a personal-machine assumption).
    """
    bootstrap_env()
    key = normalize_chain(chain or _CHAIN)
    env_key = "ALCHEMY_ETHEREUM_URL" if key == "ethereum" else "ALCHEMY_BASE_URL"
    u = os.environ.get(env_key)
    if u:
        return u.strip().strip('"')
    if key == "base":
        eth = os.environ.get("ALCHEMY_ETHEREUM_URL", "").strip().strip('"')
        if "eth-mainnet.g.alchemy.com" in eth:
            return eth.replace("eth-mainnet.g.alchemy.com", "base-mainnet.g.alchemy.com")
    raise SystemExit(f"{env_key} is not set (copy .env.example to .env and fill it in)")


def _set_chain(chain: str) -> None:
    global _CHAIN, _CHAIN_ID, _RPC
    _CHAIN = normalize_chain(chain)
    _CHAIN_ID = int(chain_config(_CHAIN)["chain_id"])
    _RPC = None


def configure_rpc_sleep(seconds: float) -> None:
    """Optional pause after each successful RPC (rate-limit hygiene)."""
    global _RPC_SLEEP
    env = os.environ.get("AMM_RPC_SLEEP", "").strip()
    if seconds > 0:
        _RPC_SLEEP = max(0.0, float(seconds))
    elif env:
        _RPC_SLEEP = max(0.0, float(env))
    else:
        _RPC_SLEEP = 0.0


def _rpc_error_rate_limited(err: object) -> bool:
    if not isinstance(err, dict):
        return False
    code = err.get("code")
    msg = str(err.get("message") or err).lower()
    if code in (429, -32005, -32016):
        return True
    return any(
        s in msg
        for s in (
            "rate limit",
            "too many requests",
            "exceeded its compute units",
            "capacity",
        )
    )


def _exception_rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "rate limit" in msg


def _rpc_backoff_seconds(attempt: int, *, rate_limited: bool) -> float:
    if rate_limited:
        base = float(os.environ.get("AMM_RPC_RATE_LIMIT_SLEEP", "30"))
        cap = float(os.environ.get("AMM_RPC_RATE_LIMIT_CAP", "600"))
        return min(cap, base * (2**attempt)) + random.uniform(0, 5)
    return 1.5 * (attempt + 1)


def rpc(method: str, params: list, retries: int | None = None):
    global _RPC
    if _RPC is None:
        _RPC = rpcurl()
    if retries is None:
        retries = int(os.environ.get("AMM_RPC_RETRIES", "16"))
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    ).encode()
    for attempt in range(retries):
        rate_limited = False
        try:
            req = urllib.request.Request(
                _RPC, data=body, headers={"Content-Type": "application/json"}
            )
            out = json.loads(urllib.request.urlopen(req, timeout=120).read())
            if "error" in out:
                if _rpc_error_rate_limited(out["error"]):
                    rate_limited = True
                raise RuntimeError(str(out["error"])[:200])
            if _RPC_SLEEP > 0:
                time.sleep(_RPC_SLEEP)
            return out["result"]
        except Exception as exc:
            rate_limited = rate_limited or _exception_rate_limited(exc)
            if attempt == retries - 1:
                raise
            wait = _rpc_backoff_seconds(attempt, rate_limited=rate_limited)
            kind = "rate_limit" if rate_limited else "rpc"
            print(
                f"rpc {kind} retry {attempt + 1}/{retries} "
                f"after {wait:.1f}s ({exc})",
                flush=True,
            )
            time.sleep(wait)


def i256(h: str) -> str:
    v = int(h, 16)
    if v >= 2**255:
        v -= 2**256
    return str(v)


def u256(h: str) -> str:
    return str(int(h, 16))


def i24(topic: str) -> int:
    v = int(topic, 16) & ((1 << 24) - 1)
    return v - 2**24 if v >= 2**23 else v


def addr_from_topic(topic: str) -> str:
    return ("0x" + topic[-40:]).lower()


def decode_event(lg: dict) -> dict | None:
    topics = [t.lower() for t in lg.get("topics") or []]
    if not topics:
        return None
    name = TOPIC_TO_EVENT.get(topics[0])
    if name is None:
        return None
    data = lg.get("data", "0x")
    if data.startswith("0x"):
        d = data[2:]
    else:
        d = data
    decoded: dict
    if name == "Swap":
        decoded = {
            "sender": addr_from_topic(topics[1]) if len(topics) > 1 else None,
            "recipient": addr_from_topic(topics[2]) if len(topics) > 2 else None,
            "amount0": i256(d[0:64]),
            "amount1": i256(d[64:128]),
            "sqrtPriceX96": u256(d[128:192]),
            "liquidity": u256(d[192:256]),
            "tick": i24("0x" + d[256:320]) if len(d) >= 320 else None,
        }
    elif name == "Mint":
        decoded = {
            "sender": "0x" + d[24:64] if len(d) >= 64 else None,
            "owner": addr_from_topic(topics[1]) if len(topics) > 1 else None,
            "tickLower": i24(topics[2]) if len(topics) > 2 else None,
            "tickUpper": i24(topics[3]) if len(topics) > 3 else None,
            "amount": u256(d[64:128]),
            "amount0": u256(d[128:192]),
            "amount1": u256(d[192:256]),
        }
    elif name == "Burn":
        decoded = {
            "owner": addr_from_topic(topics[1]) if len(topics) > 1 else None,
            "tickLower": i24(topics[2]) if len(topics) > 2 else None,
            "tickUpper": i24(topics[3]) if len(topics) > 3 else None,
            "amount": u256(d[0:64]),
            "amount0": u256(d[64:128]),
            "amount1": u256(d[128:192]),
        }
    elif name == "Collect":
        decoded = {
            "owner": addr_from_topic(topics[1]) if len(topics) > 1 else None,
            "recipient": "0x" + d[24:64] if len(d) >= 64 else None,
            "tickLower": i24(topics[2]) if len(topics) > 2 else None,
            "tickUpper": i24(topics[3]) if len(topics) > 3 else None,
            "amount0": u256(d[64:128]),
            "amount1": u256(d[128:192]),
        }
    elif name == "Initialize":
        decoded = {
            "sqrtPriceX96": u256(d[0:64]),
            "tick": i24("0x" + d[64:128]) if len(d) >= 128 else None,
        }
    else:
        return None
    return {
        "event_name": name,
        "decoded": decoded,
        "raw_topics": lg.get("topics") or [],
        "raw_data": lg.get("data") or "0x",
    }


def next_part_index(event_root: Path) -> int:
    mx = -1
    for f in event_root.rglob("part-*.parquet"):
        try:
            n = int(f.stem.split("-", 1)[1])
            mx = max(mx, n)
        except ValueError:
            continue
    return mx + 1


def check_block_hash_consistency(rows: list[dict]) -> None:
    """Cheap, same-batch reorg-contamination check (CODE_REVIEW.md repair
    on reorg/finality hardening): every log for a given block_number in
    this batch must report the same block_hash. This does not detect a
    reorg that happened entirely between two separate `pull_range` calls
    (that would require re-querying already-written blocks against the
    current chain head, out of scope for this bounded pass -- see
    FINALITY_CONTRACT below) -- it only catches the case where a single
    eth_getLogs response batch itself straddled a reorg mid-call, which is
    otherwise silent because the dedup key (block_number, transaction_index,
    log_index) does not include block_hash."""
    seen: dict[int, str] = {}
    for r in rows:
        bn, bh = r["block_number"], r["block_hash"]
        prior = seen.get(bn)
        if prior is not None and prior != bh:
            raise RuntimeError(
                f"block_hash mismatch within one write batch for block {bn}: "
                f"{prior!r} vs {bh!r} -- likely a reorg mid-batch; refusing to "
                f"write potentially inconsistent rows"
            )
        seen[bn] = bh


FINALITY_CONTRACT = """
Ingestion only requests blocks up to (chain tip - finality_lag), default 64
blocks (run(finality_lag=64)) -- deep enough that an unnoticed reorg
reaching an already-ingested block is not expected in practice for
Ethereum/Base, but is not formally impossible. Residual risk, documented
rather than solved in this pass: this pipeline does not re-verify an
already-written block's hash against the current canonical chain on a
later run (that would require an extra eth_getBlockByNumber per historical
block on every rerun, which is disproportionate for a research-replication
pipeline). check_block_hash_consistency() catches the narrower case of a
reorg occurring mid-eth_getLogs-batch. A rerun of pull_range()/run() is
deterministic and idempotent for the range it covers: parts are
append-only and never rewritten, and downstream reconstruction dedups by
(block_number, transaction_index, log_index) keeping the latest
extraction_run -- so a rerun after a suspected reorg can safely re-pull
the affected range and the newer extraction_run's rows will be the ones
kept.
"""


def write_part(rows: list[dict], extraction_run: str, *, chain: str) -> Path | None:
    """Append a new hive part; never overwrites existing files."""
    if not rows:
        return None
    check_block_hash_consistency(rows)
    ensure_universal_dirs(chain=chain)
    part_root = pool_events_partition_dir(chain=chain)
    by_key: dict[tuple[str, int, int], list[dict]] = {}
    for r in rows:
        ts = int(r["block_timestamp"])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        key = (r["event_name"], dt.year, dt.month)
        by_key.setdefault(key, []).append(r)

    written: list[Path] = []
    for (event, year, month), chunk in by_key.items():
        dest_dir = (
            part_root / f"event={event}" / f"year={year:04d}" / f"month={month:02d}"
        )
        dest_dir.mkdir(parents=True, exist_ok=True)
        idx = next_part_index(part_root / f"event={event}")
        path = dest_dir / f"part-{idx:07d}.parquet"
        while path.exists():
            idx += 1
            path = dest_dir / f"part-{idx:07d}.parquet"
        _write_parquet(path, chunk)
        written.append(path)
        print(f"wrote {path} rows={len(chunk)} run={extraction_run}", flush=True)
    return written[0] if written else None


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import duckdb
    import pandas as pd

    df = pd.DataFrame(rows, columns=ENVELOPE_COLS)
    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        out = str(path).replace("'", "''")
        con.execute(f"COPY (SELECT * FROM df) TO '{out}' (FORMAT PARQUET)")
    finally:
        con.close()


def block_timestamp(bn: int, cache: dict[int, int], stride: int = 1) -> int:
    key = bn if stride <= 1 else (bn // stride * stride)
    if key not in cache:
        blk = rpc("eth_getBlockByNumber", [hex(key), False])
        cache[key] = int(blk["timestamp"], 16)
    return cache[key]


def log_to_row(
    lg: dict,
    *,
    ts: int,
    batch: int,
    run: str,
    chain_id: int,
) -> dict | None:
    parsed = decode_event(lg)
    if parsed is None:
        return None
    return {
        "chain_id": chain_id,
        "pool_address": lg["address"].lower(),
        "event_name": parsed["event_name"],
        "block_number": int(lg["blockNumber"], 16),
        "block_hash": lg.get("blockHash") or "",
        "block_timestamp": ts,
        "transaction_hash": lg["transactionHash"],
        "transaction_index": int(lg["transactionIndex"], 16),
        "log_index": int(lg["logIndex"], 16),
        "decoded_json": json.dumps(parsed["decoded"], separators=(",", ":")),
        "raw_topics_json": json.dumps(parsed["raw_topics"]),
        "raw_data": parsed["raw_data"],
        "extraction_batch": batch,
        "extraction_run": run,
        "timestamp_source": "rpc_block",
    }


def discover_pools_from_lake(limit: int = 0, *, chain: str) -> list[str]:
    import duckdb

    g = str(
        pool_events_partition_dir(chain=chain) / "event=Swap" / "**" / "*.parquet"
    ).replace("'", "''")
    con = duckdb.connect(database=":memory:")
    try:
        sql = f"""
            SELECT DISTINCT lower(pool_address) AS p
            FROM read_parquet('{g}', hive_partitioning=true, union_by_name=true)
            ORDER BY 1
        """
        if limit > 0:
            sql += f" LIMIT {int(limit)}"
        return [r[0] for r in con.execute(sql).fetchall()]
    finally:
        con.close()


def resolve_pools(pools_file: str | None, limit: int = 0, *, chain: str) -> list[str]:
    if pools_file:
        u = load_universe(Path(pools_file))
        if not u:
            raise SystemExit(f"no pools in {pools_file}")
        pools = sorted(u)
    else:
        pools = discover_pools_from_lake(limit=limit, chain=chain)
    if limit > 0:
        pools = pools[:limit]
    if not pools:
        raise SystemExit("no pools resolved — pass --pools-file or ensure Swap parts exist")
    return pools


def pull_range(
    *,
    pools: list[str],
    event_names: list[str],
    frm: int,
    tip: int,
    step: int = 1000,
    ts_stride: int = 1,
    estimate_only: bool = False,
    address_batch: int = 80,
    chain: str,
    chain_id: int,
) -> dict:
    topics = [TOPICS[n] for n in event_names if n in TOPICS]
    if not topics:
        raise SystemExit(f"unknown events: {event_names}")
    gap = max(0, tip - frm + 1)
    est = {
        "chain": chain,
        "chain_id": chain_id,
        "frm": frm,
        "tip": tip,
        "gap_blocks": gap,
        "n_pools": len(pools),
        "events": event_names,
        "est_getLogs": (gap + step - 1) // step if gap else 0,
        "address_batch": address_batch,
    }
    print(json.dumps(est, indent=2), flush=True)
    if estimate_only or tip < frm:
        return {**est, "done": True, "new_rows": 0}

    run = datetime.now(timezone.utc).strftime(
        f"pool_events_{chain}_%Y%m%dT%H%M%SZ"
    )
    tscache: dict[int, int] = {}
    new_rows = 0
    batch = 0
    buf: list[dict] = []

    pool_batches = [
        pools[i : i + address_batch] for i in range(0, len(pools), address_batch)
    ]

    for pbatch in pool_batches:
        cur = frm
        cur_step = step
        while cur <= tip:
            to = min(cur + cur_step - 1, tip)
            try:
                logs = rpc(
                    "eth_getLogs",
                    [
                        {
                            "fromBlock": hex(cur),
                            "toBlock": hex(to),
                            "address": pbatch,
                            "topics": [topics],
                        }
                    ],
                )
            except Exception as exc:
                # rpc() has already exhausted its own retry/backoff budget
                # (AMM_RPC_RETRIES, default 16) by the time we get here.
                if cur_step > 50:
                    # Narrow the range and retry -- this is a legitimate
                    # provider-limit adaptation, not a skip.
                    cur_step = max(50, cur_step // 2)
                    continue
                # Already at the minimum range and it still fails: FAIL
                # CLOSED (CODE_REVIEW.md repair #1). Never silently advance
                # past an unrecorded failed range -- that produces an
                # undetectable gap in the raw event lake.
                gap = {
                    "chain": chain, "chain_id": chain_id,
                    "pool_batch": pbatch, "from_block": cur, "to_block": to,
                    "error": str(exc)[:500],
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                gap_path = pool_events_partition_dir(chain=chain).parent / "manifests" / "ingestion_gaps.jsonl"
                gap_path.parent.mkdir(parents=True, exist_ok=True)
                with gap_path.open("a") as f:
                    f.write(json.dumps(gap) + "\n")
                raise RuntimeError(
                    f"eth_getLogs failed for blocks {cur}..{to} on {chain} after "
                    f"exhausting retries at the minimum 50-block step; gap recorded "
                    f"in {gap_path}. Ingestion run marked incomplete -- refusing to "
                    f"silently skip this range."
                ) from exc

            batch += 1
            for lg in logs:
                bn = int(lg["blockNumber"], 16)
                ts = block_timestamp(bn, tscache, stride=ts_stride)
                row = log_to_row(lg, ts=ts, batch=batch, run=run, chain_id=chain_id)
                if row is None:
                    continue
                if row["event_name"] not in event_names:
                    continue
                buf.append(row)
                new_rows += 1

            if len(buf) >= 50_000:
                write_part(buf, run, chain=chain)
                buf = []

            cur = to + 1
            if cur_step < step:
                cur_step = min(step, cur_step + 100)
            if batch % 20 == 0:
                print(
                    f"chunk={batch} block={cur}/{tip} pools_batch={len(pbatch)} "
                    f"new_rows={new_rows}",
                    flush=True,
                )

    if buf:
        write_part(buf, run, chain=chain)
    return {**est, "done": True, "new_rows": new_rows, "extraction_run": run}


def pool_max_blocks(event_name: str = WATERMARK_EVENT, *, chain: str) -> dict[str, int]:
    cov = load_coverage(chain=chain)
    out: dict[str, int] = {}
    for r in cov.get("pools") or []:
        if r.get("event_name") != event_name:
            continue
        if r.get("max_block") is None:
            continue
        out[r["pool_address"]] = int(r["max_block"])
    return out


def run(
    *,
    estimate_only: bool = False,
    max_blocks: int = 0,
    step: int = 1000,
    ts_stride: int = 1,
    finality_lag: int = 64,
    from_block: int = 0,
    to_block: int = 0,
    to_watermark: int = 0,
    events: str = "Swap,Mint,Burn,Collect",
    pools_file: str | None = None,
    pool_limit: int = 0,
    chain: str = "ethereum",
    address_batch: int = 80,
    rpc_sleep: float = 0.0,
) -> dict:
    """Pull events into hive parts.

    Modes:
    - default tip catch-up: from coverage watermark+1 (or from_block) to chain tip-lag
    - --to-watermark W: per-pool continuous backfill from each pool's max+1 to W
    """
    chain = normalize_chain(chain)
    _set_chain(chain)
    configure_rpc_sleep(rpc_sleep)
    ensure_universal_dirs(chain=chain)
    event_names = [e.strip() for e in events.split(",") if e.strip()]
    pools = resolve_pools(pools_file, limit=pool_limit, chain=chain)

    if to_watermark > 0:
        return run_to_watermark(
            pools=pools,
            event_names=event_names,
            watermark=to_watermark,
            step=step,
            ts_stride=ts_stride,
            estimate_only=estimate_only,
            chain=chain,
            address_batch=address_batch,
        )

    latest = int(rpc("eth_blockNumber", []), 16)
    tip = to_block if to_block > 0 else (latest - finality_lag)
    if max_blocks > 0 and from_block > 0:
        tip = min(tip, from_block + max_blocks - 1)

    if from_block > 0:
        frm = from_block
    else:
        cov = load_coverage(chain=chain)
        wm = cov.get("watermark_block")
        frm = int(wm) + 1 if wm is not None else tip + 1

    if max_blocks > 0:
        tip = min(tip, frm + max_blocks - 1)

    return pull_range(
        pools=pools,
        event_names=event_names,
        frm=frm,
        tip=tip,
        step=step,
        ts_stride=ts_stride,
        estimate_only=estimate_only,
        address_batch=address_batch,
        chain=chain,
        chain_id=_CHAIN_ID,
    )


def run_to_watermark(
    *,
    pools: list[str],
    event_names: list[str],
    watermark: int,
    step: int,
    ts_stride: int,
    estimate_only: bool,
    chain: str,
    address_batch: int = 80,
) -> dict:
    """Continuously backfill each pool from max_block+1 to W for the watermark event."""
    maxima = pool_max_blocks(WATERMARK_EVENT, chain=chain)
    if not maxima:
        raise SystemExit(
            "coverage has no per-pool maxima — run amm-data coverage build first"
        )

    total_new = 0
    plans = []
    for p in pools:
        mx = maxima.get(p.lower())
        if mx is None:
            plans.append({"pool": p, "skip": "no_max_in_coverage"})
            continue
        if mx >= watermark:
            plans.append({"pool": p, "skip": "already_at_or_above_W", "max": mx})
            continue
        frm = mx + 1
        plans.append({"pool": p, "frm": frm, "tip": watermark})

    need = [x for x in plans if "frm" in x]
    print(
        json.dumps(
            {
                "mode": "to_watermark",
                "chain": chain,
                "watermark": watermark,
                "n_pools": len(pools),
                "n_need_backfill": len(need),
                "events": event_names,
            },
            indent=2,
        ),
        flush=True,
    )
    if estimate_only:
        return {"done": True, "plans": plans, "new_rows": 0}
    if not need:
        return {"done": True, "plans": plans, "new_rows": 0}

    groups: dict[int, list[str]] = {}
    for item in need:
        groups.setdefault(int(item["frm"]), []).append(item["pool"])

    for frm, group in sorted(groups.items()):
        result = pull_range(
            pools=group,
            event_names=event_names,
            frm=frm,
            tip=watermark,
            step=step,
            ts_stride=ts_stride,
            estimate_only=False,
            address_batch=address_batch,
            chain=chain,
            chain_id=_CHAIN_ID,
        )
        total_new += int(result.get("new_rows") or 0)

    return {"done": True, "watermark": watermark, "new_rows": total_new, "plans": plans}
