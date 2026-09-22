"""Single path-resolution module for research-core (repair queue item 4/5).

ALL functional (runtime-required) path resolution in research-core goes
through this module. No other module should hardcode an absolute path or
independently read a path-related environment variable.

Repository-relative, git-safe locations are computed from this file's own
position (`REPO_ROOT`), never from a hardcoded machine path.

External, non-vendored data (the response-panel cache built by a prior
pipeline run, the raw multi-year AMM event lake, block-header CSVs) is
**not part of this repository** and cannot have a working default on a
fresh clone. For that data, this module provides a single environment
variable with a safe, empty-by-default fallback under `data/`:

    RESEARCH_CORE_DATA_ROOT          -- local raw/interim/derived data root
                                         (default: <repo>/data)
    RESEARCH_CORE_EXTERNAL_CACHE     -- root of externally-supplied derived
                                         artifacts (response panels, block
                                         headers, RMT precursor panels) that
                                         this repo does not itself produce
                                         or redistribute (default:
                                         <repo>/data/external_cache)
    RESEARCH_CORE_RAW_AMM_ROOT       -- root of the raw multi-year on-chain
                                         event lake (default:
                                         <repo>/data/raw/pool_events)

On a fresh clone none of these directories will contain data, and every
function here returns a path that simply won't exist yet -- callers (tests,
scripts) are responsible for checking existence and skipping/failing
accordingly (see tests/integration/*.py for the pattern), not this module.

Provenance: the `CHAINS` table and the pool-events hive-layout helpers are a
minimal vendored subset of `amm_data.chains` / `amm_data.paths`
(`/Users/joseph/amm-data/src/amm_data/{chains,paths}.py`), copied rather
than imported so research-core has no runtime dependency on the separate,
unpublished `amm-data` package (CODE_REVIEW.md finding, repair #2).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def bootstrap_env() -> None:
    """Load <repo>/.env once (idempotent via os.environ.setdefault)."""
    _load_dotenv(REPO_ROOT / ".env")


def _env_path(var: str, default: Path) -> Path:
    bootstrap_env()
    raw = os.environ.get(var, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return default


def data_root() -> Path:
    return _env_path("RESEARCH_CORE_DATA_ROOT", REPO_ROOT / "data")


def raw_root() -> Path:
    return data_root() / "raw"


def interim_root() -> Path:
    return data_root() / "interim"


def derived_root() -> Path:
    return data_root() / "derived"


def outputs_root() -> Path:
    return REPO_ROOT / "outputs" / "tables"


def external_cache_root() -> Path:
    """Root of externally-supplied, non-vendored derived artifacts (the
    response-panel cache, precursor RMT panels, block-header CSVs). This
    repo does not populate this directory; a user pointing it at an
    existing artifact tree should preserve the original relative layout
    (jfqa_service_confirmatory_long/panels/, jfqa_hf_response_full/derived/,
    jfqa_service_primitives/derived/, ...), matching the structure each
    accessor function below expects."""
    return _env_path("RESEARCH_CORE_EXTERNAL_CACHE", data_root() / "external_cache")


def raw_amm_event_lake_root() -> Path:
    """Root of the raw, multi-year decoded on-chain event lake (Swap/Mint/
    Burn/Collect/Initialize parquet). Defaults to this repo's own raw/
    output directory (i.e. what src/research_core/ingestion/eth_pool_events.py
    itself writes), so a from-scratch ingest run and a downstream
    reconstruction run agree on where the data lives without either
    hardcoding the other's location."""
    return _env_path("RESEARCH_CORE_RAW_AMM_ROOT", raw_root() / "pool_events")


# ---------------------------------------------------------------------------
# Pool-events hive layout (vendored subset of amm_data.chains / amm_data.paths)
# ---------------------------------------------------------------------------

CHAINS: dict[str, dict] = {
    "ethereum": {
        "chain_id": 1,
        "pool_events_relpath": "pool_events",
        "glob_suffix": "event=*/**/*.parquet",
        "coverage_file": "pool_events_coverage_v1.json",
    },
    "base": {
        "chain_id": 8453,
        "pool_events_relpath": "pool_events/base",
        "glob_suffix": "chain_id=8453/event=*/**/*.parquet",
        "coverage_file": "pool_events_coverage_v1_base.json",
    },
}


def normalize_chain(name: str) -> str:
    key = (name or "ethereum").strip().lower()
    if key not in CHAINS:
        raise SystemExit(f"unknown chain {name!r}; expected one of: {', '.join(sorted(CHAINS))}")
    return key


def chain_config(name: str) -> dict:
    return CHAINS[normalize_chain(name)]


def pool_events_dir(chain: str = "ethereum") -> Path:
    cfg = chain_config(chain)
    return raw_amm_event_lake_root().parent / cfg["pool_events_relpath"]


def pool_events_partition_dir(chain: str = "ethereum") -> Path:
    d = pool_events_dir(chain)
    if normalize_chain(chain) == "base":
        d = d / f"chain_id={chain_config(chain)['chain_id']}"
    return d


def coverage_manifest_path(chain: str = "ethereum") -> Path:
    cfg = chain_config(chain)
    return raw_amm_event_lake_root().parent / "manifests" / cfg["coverage_file"]


def ensure_universal_dirs(chain: str = "ethereum") -> Path:
    root = raw_amm_event_lake_root().parent
    for rel in (chain_config(chain)["pool_events_relpath"], "manifests"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Externally-supplied derived artifacts (response panels, headers, etc.)
# ---------------------------------------------------------------------------

def confirmatory_panels_dir() -> Path:
    """eth5/base5/base1_pre_response_panel.parquet, *_pre_panel.parquet,
    *_pre_block_headers.csv -- built by the (currently external, not
    re-run in this repo) confirmatory reconstruction pipeline."""
    return external_cache_root() / "jfqa_service_confirmatory_long" / "panels"


def confirmatory_section_e_raw_json() -> Path:
    return external_cache_root() / "jfqa_service_confirmatory_long" / "section_e_raw.json"


def hf_response_full_derived_dir() -> Path:
    return external_cache_root() / "jfqa_hf_response_full" / "derived"


def service_primitives_derived_dir() -> Path:
    return external_cache_root() / "jfqa_service_primitives" / "derived"


def service_primitives_block_headers_dir() -> Path:
    return external_cache_root() / "jfqa_service_primitives" / "block_headers"


def public_reference_audit_block_headers_dir() -> Path:
    return external_cache_root() / "jfqa_public_reference_audit" / "block_headers"


def rmt_frozen_table_csv() -> Path:
    """Frozen table produced by a prior scripts/run_rmt.py run; used as the
    self-verification gate by scripts/run_support_ladder.py. Falls back to
    the checked-in test fixture copy if no external cache is configured."""
    external = external_cache_root() / "jfqa_response_map_transport" / "RESPONSE_MAP_TRANSPORT_TABLE.csv"
    if external.is_file():
        return external
    return REPO_ROOT / "tests" / "fixtures" / "frozen_rmt_table.csv"


def rmt_frozen_bootstrap_summary_csv() -> Path:
    external = external_cache_root() / "jfqa_response_map_transport" / "RESPONSE_MAP_TRANSPORT_BOOTSTRAP_SUMMARY.csv"
    if external.is_file():
        return external
    return REPO_ROOT / "tests" / "fixtures" / "frozen_rmt_bootstrap_summary.csv"


def legacy_bridge_eth5_parquet() -> Path | None:
    """Optional legacy-lineage columns for ETH5 (deprecated pilot,
    LEGACY_ORPHAN per CODE_CORE_EXTRACTION_AUDIT.md). Returns None (not a
    hardcoded personal path) unless explicitly configured -- the caller
    already treats a missing/None path as "no legacy columns", which is
    the correct behavior for a from-scratch build."""
    raw = os.environ.get("RESEARCH_CORE_LEGACY_BRIDGE_PARQUET", "").strip()
    return Path(raw).expanduser().resolve() if raw else None
