# Canonical response-panel schema

Produced by `scripts/build_panel.py` (from `build_response_panel_pre.py`),
consumed by `scripts/run_confirmation.py`, `run_rmt.py`,
`run_support_ladder.py`, `run_decomposition.py`.

One row per (pool, block).

| Column | Type | Meaning |
|---|---|---|
| `block_timestamp` | int64 (unix seconds) | Block timestamp |
| `xstar_hf_log` | float64 | `log X*` composite reference at this block (strict-prior, ≤3s, Bybit+OKX equal-weight log midpoint) |
| `d_pre_hf` | float64 | `log(P_AMM,pre) - log(X*)` at the block's pre-state |
| `d_post_hf` | float64 | `log(P_AMM,post) - log(X*)` at the block's post-state |
| `x` | float64 | `d_pre_hf / gamma_fee` (see `src/research_core/methods/estimand.py::gamma_fee`) |
| `correction_hf` | float64 | `C = d_pre_hf - d_post_hf` |
| `has_focal_swap` | bool | Focal-pool swap indicator `S` |
| `corrective_hf` | bool | `1{|d_post_hf| < |d_pre_hf|}` |
| `retained_hf` | bool | Passed the reference-freshness gate (both venues ≤3s) |
| `x_bin` | categorical | One of the 9 frozen support bins: `[0,.5) [.5,.75) [.75,1) [1,1.25) [1.25,1.5) [1.5,2) [2,3) [3,5) [5,inf)` |

Estimation scripts filter to `retained_hf & has_focal_swap` before fitting
the hinge regression (`x`, `correction_hf` on the S=1 subsample); the
decomposition scripts use the full `retained_hf` sample (both S=0 and S=1)
to compute `P(S=1|x)` per bin.

Upstream inputs to this panel:
- AMM block state: `src/research_core/reconstruction/assemble_gold_panel.py`
  (from `amm_replay_engine.py` + raw Swap/Mint/Burn events)
- External reference: `scripts/build_reference.py` (from
  `reconstruct_bybit_bbo.py` + `reconstruct_okx_bbo.py`)
