"""POST-CONFIRMATORY EXPLORATORY ROBUSTNESS; frozen RMT estimators reused."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

# Originally: sys.path.insert(0, "../../jfqa_response_map_transport/code")
# then `from rmt_core import ...` -- that relative layout no longer applies
# once both files live in the same research_core.methods package, so this
# now imports the sibling module directly. Same symbols, same module
# (src/research_core/methods/rmt.py is rmt_core.py, copied verbatim).
from research_core.methods.rmt import (aggregate_day_stats, transport_error, standardized_difference,
                      shapley_decomposition, percentile_ci, BP, RMT_REPS, RMT_SEED)

# WORK/RMT: originally the work/jfqa_response_map_transport_support and
# work/jfqa_response_map_transport directories (driver-script output /
# frozen-fixture roots respectively). Re-pointed at code's own
# outputs/tests-fixtures dirs -- see run_support_ladder.py and
# tests/test_support_ladder.py, which read these for integration checks
# that require run_support_ladder.py to have been executed first.
_ROOT = Path(__file__).resolve().parents[3]
WORK = _ROOT / "outputs" / "tables"
RMT = _ROOT / "tests" / "fixtures"

CAPS = ((1.5, 6), (2.0, 7), (3.0, 8), (5.0, 9), (None, 10))
STATS = ('TE_D_to_H', 'STD_DIFF', 'DELTA_pi', 'DELTA_p', 'DELTA_m')
TOL = 1e-12  # native log units, numerical validation only


def capped_map(full, k):
    if k not in (6, 7, 8, 9, 10):
        raise ValueError('not a declared complete-bin cap')
    n = full.n[:k]
    if n.sum() == 0:
        raise ValueError('empty cap')
    pi = n / n.sum()
    assert abs(pi.sum() - 1) < TOL
    return SimpleNamespace(n=n, pi=pi, p=full.p[:k], m=full.m[:k])


def evaluate(md, mh, k):
    try:
        d, h = capped_map(md, k), capped_map(mh, k)
    except ValueError:
        return None
    if not all(np.isfinite(a).all() for a in (d.p, d.m, h.p, h.m)):
        return None
    te = transport_error(h.pi, d, h)
    std = standardized_difference(d, h)
    sh = shapley_decomposition(d, h)
    assert abs(sh['shapley_residual']) < TOL
    assert abs(std['pi_bar_sum'] - 1) < TOL
    return {**te, **{key: val for key, val in std.items() if key != 'pi_bar'},
            **sh, 'Mean_C_D_bp': sh['F_D'] * BP, 'Mean_C_H_bp': sh['F_H'] * BP,
            'N_D': int(d.n.sum()), 'N_H': int(h.n.sum())}


def bootstrap(daily_d, daily_h, reps=RMT_REPS, seed=RMT_SEED):
    rng = np.random.default_rng(seed)
    nd, nh = daily_d[0].shape[0], daily_h[0].shape[0]
    draws = np.full((reps, len(CAPS), len(STATS)), np.nan)
    residuals = np.full((reps, len(CAPS)), np.nan)
    mean_residuals = np.full((reps, len(CAPS), 2), np.nan)
    for i in range(reps):
        wd = np.bincount(rng.integers(0, nd, size=nd), minlength=nd).astype(float)
        wh = np.bincount(rng.integers(0, nh, size=nh), minlength=nh).astype(float)
        md = aggregate_day_stats(*daily_d, wd)
        mh = aggregate_day_stats(*daily_h, wh)
        for j, (_, k) in enumerate(CAPS):
            result = evaluate(md, mh, k)
            if result is None:
                continue
            residuals[i, j] = result['shapley_residual']
            for widx, (daily, weights, key) in enumerate(((daily_d, wd, 'Mean_C_D_bp'), (daily_h, wh, 'Mean_C_H_bp'))):
                direct = (weights @ daily[2][:, :k]).sum() / (weights @ daily[0][:, :k]).sum()
                mean_residuals[i, j, widx] = direct - result[key] / BP
                assert abs(mean_residuals[i, j, widx]) < TOL
            draws[i, j] = [result[s] * BP for s in STATS]
    return draws, residuals, mean_residuals
