import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from research_core.paths import confirmatory_panels_dir  # noqa: E402

PAN = str(confirmatory_panels_dir())

DISCOVERY = {
    "ETH5": (25469764, 25692172),
    "BASE5": (48253332, 49592526),
    "BASE1": (48253332, 49592526),
}
POOLS = {
    "ETH5": {"path": f"{PAN}/eth5_pre_response_panel.parquet", "block_panel": f"{PAN}/eth5_pre_panel.parquet",
              "pool_address": "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640", "fee_fraction": 0.0005, "gamma": -np.log(1-0.0005)},
    "BASE5": {"path": f"{PAN}/base5_pre_response_panel.parquet", "block_panel": f"{PAN}/base5_pre_panel.parquet",
              "pool_address": "0xd0b53d9277642d899df5c87a3966a349a798f224", "fee_fraction": 0.0005, "gamma": -np.log(1-0.0005)},
    "BASE1": {"path": f"{PAN}/base1_pre_response_panel.parquet", "block_panel": f"{PAN}/base1_pre_panel.parquet",
              "pool_address": "0xb4cb800910b228ed3d0834cf79d697127bbb00e5", "fee_fraction": 0.0001, "gamma": -np.log(1-0.0001)},
}

BINS = [(0, .25), (.25, .5), (.5, .75), (.75, 1), (1, 1.25), (1.25, 1.5), (1.5, 2), (2, 3), (3, 5), (5, np.inf)]

results = {}
for market, cfg in POOLS.items():
    t = pq.read_table(cfg["path"]).to_pandas()
    bp = pq.read_table(cfg["block_panel"], columns=["block_number", "pool_address", "chain_id",
                                                       "pre_sqrtPriceX96", "post_sqrtPriceX96"]).to_pandas()

    r = {}
    # 1/18: no overlap with discovery window
    dmin, dmax = DISCOVERY[market]
    r["overlap_with_discovery_n"] = int(((bp.block_number >= dmin) & (bp.block_number <= dmax)).sum())

    # 2: pool address correctness
    r["pool_address_correct"] = bool((bp.pool_address.str.lower() == cfg["pool_address"]).all())

    # 3: canonical replay continuity (post_i == pre_{i+1})
    bp_sorted = bp.sort_values("block_number")
    cont_violations = int((bp_sorted.post_sqrtPriceX96.values[:-1] != bp_sorted.pre_sqrtPriceX96.values[1:]).sum())
    r["state_continuity_violations"] = cont_violations

    # 12: no focal swap => C == 0 (on retained rows)
    ret = t[t.retained_hf]
    no_swap = ret[~ret.has_focal_swap]
    r["no_swap_n"] = int(len(no_swap))
    r["no_swap_max_abs_correction"] = float(no_swap.correction_hf.abs().max()) if len(no_swap) else 0.0

    # 13: K (post closer) => S (swap occurred)
    K = (ret.d_post_hf < ret.d_pre_hf)
    S = ret.has_focal_swap
    r["K_implies_S_violations"] = int((K & ~S).sum())

    # 14/15: exact decomposition per frozen bin
    bins_out = []
    max_resid = 0.0
    for lo, hi in BINS:
        sub = ret[(ret.x >= lo) & (ret.x < hi)]
        n = len(sub)
        if n == 0:
            continue
        mean_C = sub.correction_hf.mean()
        P_S = sub.has_focal_swap.mean()
        E_C_given_S = sub.loc[sub.has_focal_swap, "correction_hf"].mean() if sub.has_focal_swap.any() else 0.0
        product = P_S * E_C_given_S
        resid = abs(mean_C - product)
        max_resid = max(max_resid, resid)
        bins_out.append({"bin": f"[{lo},{hi})", "n": n, "mean_C": mean_C, "P_S": P_S,
                          "E_C_given_S": E_C_given_S, "product": product, "residual": resid})
    r["decomposition_bins"] = bins_out
    r["decomposition_max_residual"] = max_resid

    # retention/coverage summary
    r["n_total_blocks"] = int(t.shape[0])
    r["n_retained"] = int(t.retained_hf.sum())
    r["n_swap_retained"] = int((t.retained_hf & t.has_focal_swap).sum())

    results[market] = r
    print(market, json.dumps({k: v for k, v in r.items() if k != "decomposition_bins"}, indent=2, default=str))

import pathlib
_OUT = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "tables" / "phase_h_qa_results.json"
_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(_OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)
