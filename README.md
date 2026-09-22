# Execution-Mediated Price Adjustment

Replication code for
**Price Adjustment When Updating Requires Execution: Evidence from Automated Market Makers**
(Wen-Ting Wang).

In a fee-gated AMM, the pool price does not continuously track an
external reference. Mispricing can accumulate inside the no-arbitrage
band until a swap executes and moves the pool. This paper measures that
execution-mediated adjustment on three Uniswap v3 ETH–USDC pools (ETH5,
BASE5, BASE1), comparing each focal-pool swap block to a strict-prior
composite reference `X*` built from Bybit and OKX L2 books.

The state variable is fee-normalized pre-swap distance
`x = |log(P_AMM,pre) - log(X*)| / γ`. The outcome is the within-block
correction `C = d_pre - d_post`. Because `C = 0` when no focal-pool swap
occurs, the conditional mean factors exactly as

```text
E[C|x] = P(S=1|x) · E[C|S=1,x]
```

—incidence times magnitude. A piecewise-linear hinge with breakpoint
fixed at `x = 1` (no search) is estimated on a confirmation window that
strictly precedes discovery. Cross-window change is then decomposed by
response-map transport (RMT) into composition (`Δ_π`), incidence (`Δ_p`),
and magnitude (`Δ_m`) Shapley terms. Support-cap robustness uses the
ladder `x ≤ 1.5 / 2 / 3 / 5 / full`.

This repository is the code path from raw inputs to those frozen
objects. It does not redistribute vendor archives or response-panel
caches.

## Prerequisites

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- For Mode B below: a local response-panel cache (layout documented in
  `src/research_core/paths.py`)
- For Mode C: archival Ethereum/Base RPC
  (`ALCHEMY_ETHEREUM_URL` / `ALCHEMY_BASE_URL`, or equivalent) and
  downloadable Bybit/OKX public L2 archives

Optional path overrides: `RESEARCH_CORE_DATA_ROOT`,
`RESEARCH_CORE_EXTERNAL_CACHE`, `RESEARCH_CORE_RAW_AMM_ROOT`.

## Installation

```bash
uv sync --extra dev
# or: pip install -e ".[dev]"
```

## Usage

### Mode A — tests (no external data)

```bash
pytest
```

Unit and fixture-regression tests always run. Cache-dependent
integration tests skip if `RESEARCH_CORE_EXTERNAL_CACHE` is unset.

### Mode B — reproduce paper tables from a cached panel

```bash
export RESEARCH_CORE_EXTERNAL_CACHE=/path/to/your/cache

python scripts/run_confirmation.py      # hinge + bootstrap (seed 20260829)
python scripts/run_rmt.py               # RMT + Shapley (seed 20260907)
python scripts/run_support_ladder.py    # RMT-side support caps
python scripts/run_support_ladder_hinge.py --panel <path> --market <M> --out-json <path>
```

Outputs write under `outputs/tables/`. Frozen numerical targets live in
`tests/fixtures/frozen_*`.

### Mode C — rebuild panels from source

```bash
# Replay AMM state from chain events
python scripts/reconstruct_amm.py \
  --pool-key ETH5 --headers <csv> --output-parquet <path>

# Strict-prior reference X* from Bybit + OKX BBO
python scripts/build_reference.py \
  --bybit-parquet <path> --okx-parquet <path> \
  --output-parquet <path> --output-manifest <path>

# Join AMM state to X*; compute x and C
python scripts/build_panel.py \
  --block-panel <path> --pool-key ETH5 \
  --xstar-hf-parquet <path> --fee-fraction 0.0005 \
  --output-parquet <path> --output-manifest <path>
```

Archive acquisition:
`src/research_core/ingestion/download_reference_archives.py`.
RPC event pull: `src/research_core/ingestion/eth_pool_events.py`.

Pool addresses, reference construction, and sample windows are in
`config/`. Methods live under `src/research_core/methods/`.

**Data rights.** Chain data is public (you supply RPC). Bybit/OKX
archives are public; this repo ships acquisition code only—raw archives
and response-panel caches are not committed.

## Citation

```bibtex
@misc{wang2026execution,
  title        = {Price Adjustment When Updating Requires Execution:
                  Evidence from Automated Market Makers},
  author       = {Wang, Wen-Ting},
  year         = {2026},
  note         = {arXiv preprint (link forthcoming)},
  howpublished = {\url{https://arxiv.org/abs/XXXX.XXXXX}}
}
```

arXiv link will be updated once the preprint is public.
