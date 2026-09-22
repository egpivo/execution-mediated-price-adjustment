#!/usr/bin/env python3
"""One fixed support-cap ladder. No frozen artifact or manuscript writes.

Import/path note: originally same-directory flat imports
(`from support_core import WORK, RMT, ...`) where WORK/RMT were paths
derived from the work/jfqa_response_map_transport(_support) directory
layout. That layout doesn't exist in code, so WORK/RMT are
redefined below to point at code's own tests/fixtures (frozen
verification target) and outputs/tables (this script's own output) --
same role, new location. Estimation logic (CAPS, evaluate, bootstrap) is
untouched.
"""
import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from research_core.methods.support_ladder import (CAPS, STATS, BP, RMT_REPS, RMT_SEED,
                          aggregate_day_stats, evaluate, bootstrap, percentile_ci, TOL)
from run_rmt import load_retained, compute_point_maps, parse_table8, verify_table8, MANUSCRIPT_TABLE8
from research_core.methods.rmt import assign_bins

_ROOT = Path(__file__).resolve().parents[1]
WORK = _ROOT / "outputs" / "tables"
RMT = _ROOT / "tests" / "fixtures"
WORK.mkdir(parents=True, exist_ok=True)
(WORK / "derived").mkdir(parents=True, exist_ok=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_full(row, frozen):
    mapping = {'Mean_C_D_bp': 'Mean_C_Discovery_bp', 'Mean_C_H_bp': 'Mean_C_Confirmation_bp',
               'C_H_from_D_bp': 'Transported_D_on_H_bp', 'C_D_std_bp': 'Std_D_bp', 'C_H_std_bp': 'Std_H_bp'}
    mapping.update({s+'_bp': s+'_bp' for s in STATS})
    for s in STATS:
        prefix = 'TE' if s == 'TE_D_to_H' else s
        for end in ('lo', 'hi'):
            mapping[s+'_CI_'+end+'_bp'] = prefix+'_CI_'+end+'_bp'
    errors = {key: abs(float(row[key]) - float(frozen[col])) for key, col in mapping.items() if key in row}
    if not all(np.isfinite(v) and v <= 1e-10 for v in errors.values()):
        raise RuntimeError('STOP: Full reproduction failed: '+str(errors))
    return max(errors.values())


def _git_head():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                        cwd=str(Path(__file__).resolve().parent)).strip()
    except Exception:
        return None


def main():
    frozen = {r['Market']: r for r in csv.DictReader((RMT/'frozen_rmt_table.csv').open())}
    frozen_boot = {(r['Market'],r['statistic']):r for r in csv.DictReader((RMT/'frozen_rmt_bootstrap_summary.csv').open())}
    manifest = {'status': 'POST-CONFIRMATORY EXPLORATORY ROBUSTNESS', 'seed': RMT_SEED,
                'reps': RMT_REPS, 'rng': 'numpy.random.default_rng; reset per market; D then H',
                'numpy': np.__version__, 'python': platform.python_version(),
                'git_head': _git_head(),
                'inputs': {}, 'diagnostics': {}, 'full_reproduction': {}, 'protected_unchanged': None}
    _methods = Path(__file__).resolve().parents[1] / 'src' / 'research_core' / 'methods'
    for f in [*sorted(Path(__file__).resolve().parent.glob('*.py')), *sorted(_methods.glob('*.py')),
              RMT/'frozen_rmt_table.csv', RMT/'frozen_rmt_bootstrap_summary.csv']:
        manifest['inputs'][str(f.resolve())] = sha(f)
    rows, bootrows = [], []
    ref_table = parse_table8(MANUSCRIPT_TABLE8)
    for market in ('ETH5', 'BASE5'):
        stats, maps, diag = {}, {}, {}
        for window in ('discovery','confirmation'):
            df, path = load_retained(market, window)
            manifest['inputs'][str(path)] = sha(path)
            days, n, ns, sc, full = compute_point_maps(df)
            assert not verify_table8(market,window,full,ref_table)
            assert (df['x'] >= 0).all()
            assert (df.loc[~df['has_focal_swap'], 'correction_hf'] == 0).all()
            stats[window] = (n,ns,sc); maps[window] = full
            np.savez_compressed(WORK/'derived'/f'{market}_{window}_daily.npz', days=days,n=n,ns=ns,sc=sc)
            xd = df['x'].to_numpy(); cd = df['correction_hf'].to_numpy(); bins = assign_bins(xd)
            direct = {}
            for cap,k in CAPS:
                mask = bins < k
                assert np.array_equal(mask,np.ones(len(df),dtype=bool) if cap is None else xd<cap)
                direct[str(cap)] = {'N':int(mask.sum()),'mean_C_bp':float(cd[mask].mean()*BP),
                                    'boundary_equal_count':0 if cap is None else int((xd==cap).sum())}
            diag[window] = {'N':len(df),'days':len(days),'first_day':str(np.datetime64(int(days.min()),'D')),
                            'last_day':str(np.datetime64(int(days.max()),'D')),'caps':direct}
            assert len(days) == (30 if window=='discovery' else 27)
            bounds = ('2026-07-06','2026-08-05') if window=='discovery' else ('2026-06-09','2026-07-05')
            assert bounds[0] <= diag[window]['first_day'] <= diag[window]['last_day'] <= bounds[1]
            print(market,window,len(df),'rows; cap equality counts', [direct[str(c)]['boundary_equal_count'] for c,k in CAPS],flush=True)
        points = [evaluate(maps['discovery'],maps['confirmation'],k) for cap,k in CAPS]
        assert all(p is not None for p in points)
        check_full(points[-1], frozen[market])  # hard gate before bootstrap
        draws, residuals, mean_residuals = bootstrap(stats['discovery'],stats['confirmation'])
        np.savez_compressed(WORK/'derived'/f'{market}_bootstrap.npz',draws_bp=draws,
                            shapley_residual_native=residuals,mean_residual_native=mean_residuals)
        for ci, ((cap,k),pt) in enumerate(zip(CAPS,points)):
            label = 'Full' if cap is None else f'x<={cap:g}'
            row = {'Market':market,'Support_cap':label,'retained_bins':k,
                   'N_D':pt['N_D'],'N_H':pt['N_H'],'Mean_C_D_bp':pt['Mean_C_D_bp'],'Mean_C_H_bp':pt['Mean_C_H_bp'],
                   'C_H_from_D_bp':pt['C_H_from_D_bp'],'C_D_std_bp':pt['C_D_std_bp'],'C_H_std_bp':pt['C_H_std_bp']}
            for window,key in [('discovery','D'),('confirmation','H')]:
                direct=diag[window]['caps'][str(cap)]
                assert row['N_'+key]==direct['N']
                assert abs(row['Mean_C_'+key+'_bp']-direct['mean_C_bp']) < TOL*BP
            invalid=[]
            for si,s in enumerate(STATS):
                sample=draws[:,ci,si]; lo,hi=percentile_ci(sample)
                nv=int(np.isfinite(sample).sum()); share=1-nv/RMT_REPS; invalid.append(share)
                row.update({s+'_bp':pt[s]*BP,s+'_CI_lo_bp':lo,s+'_CI_hi_bp':hi})
                bootrows.append({'Market':market,'Support_cap':label,'statistic':s,'point_bp':pt[s]*BP,
                                 'ci_lo_bp':lo,'ci_hi_bp':hi,'n_valid':nv,'n_invalid':RMT_REPS-nv,
                                 'invalid_share':share,'BOOTSTRAP_SUPPORT_INSTABILITY':share>0.05})
                if cap is None:
                    fb=frozen_boot[market,s]
                    assert nv==int(fb['n_valid'])
                    for value,col in [(pt[s]*BP,'point_bp'),(lo,'ci_lo_bp'),(hi,'ci_hi_bp')]:
                        assert abs(value-float(fb[col]))<=1e-10,'STOP: frozen bootstrap summary mismatch'
            row['invalid_bootstrap_share_max']=max(invalid)
            row['BOOTSTRAP_SUPPORT_INSTABILITY']=max(invalid)>0.05
            row['shapley_residual_native']=pt['shapley_residual']
            row['max_bootstrap_shapley_residual_native']=float(np.nanmax(np.abs(residuals[:,ci]))) if np.isfinite(residuals[:,ci]).any() else None
            rows.append(row)
        manifest['full_reproduction'][market]={'status':'PASS','max_abs_error_bp':check_full(rows[-1],frozen[market])}
        diag['max_bootstrap_mean_residual_native']=float(np.nanmax(np.abs(mean_residuals)))
        diag['max_bootstrap_shapley_residual_native']=float(np.nanmax(np.abs(residuals)))
        manifest['diagnostics'][market]=diag
        print(market,'Full reproduction PASS',flush=True)
    # Only publish summary tables after both markets pass the frozen Full gates.
    write_csv(WORK/'RESPONSE_MAP_TRANSPORT_SUPPORT_TABLE.csv',rows)
    write_csv(WORK/'RESPONSE_MAP_TRANSPORT_SUPPORT_BOOTSTRAP.csv',bootrows)
    # Originally compared against a 'protected_before.json' snapshot written by
    # an earlier orchestration step not reproduced in this extraction (see
    # CODE_CORE_EXTRACTION_PLAN.md). Skip rather than fabricate a pass.
    protected_before = WORK/'derived/protected_before.json'
    if protected_before.is_file():
        before=json.loads(protected_before.read_text())
        changed=[f for f,digest in before.items() if sha(f)!=digest]
        assert not changed,changed
        manifest['protected_unchanged']=len(before)
    else:
        manifest['protected_unchanged']='SKIPPED: no protected_before.json snapshot in this extraction'
    (WORK/'derived/run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Complete: 10 cap rows, 50 bootstrap summaries; protected_unchanged=',manifest['protected_unchanged'])

if __name__=='__main__':main()
