"""Bounded invariants; empirical checks read stored sufficient statistics only.

CODE_REVIEW.md repair #8: the two artifact-dependent tests below previously
read from `outputs/tables/derived/` (git-ignored, ephemeral -- only
populated by a manual `run_support_ladder.py` run) and would raise
`FileNotFoundError`, not skip, on a fresh clone. Small (<1MB total)
immutable copies of those artifacts are now checked into
`tests/fixtures/support_ladder_derived/` and `tests/fixtures/`, so these
are genuine fixture-regression tests, not disguised integration tests."""
import csv
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from support_core import (WORK,RMT,CAPS,STATS,BP,capped_map,evaluate,aggregate_day_stats)
from rmt_core import maps_from_counts,assign_bins
from run_support import check_full

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DERIVED_FIXTURES = FIXTURES / "support_ladder_derived"


def synthetic(n, p=None, m=None):
    n=np.asarray(n,dtype=float)
    p=np.full(10,.5) if p is None else p
    m=np.linspace(-1,8,10)*1e-4 if m is None else m
    return maps_from_counts(n,n*p,n*p*m)


@pytest.mark.parametrize('cap,k',CAPS)
def test_weights_means_and_unchanged_map(cap,k):
    d=synthetic(np.arange(1,11)*100)
    h=synthetic(np.arange(10,0,-1)*100)
    dc,hc=capped_map(d,k),capped_map(h,k)
    assert dc.pi.sum()==pytest.approx(1,abs=1e-14)
    assert hc.pi.sum()==pytest.approx(1,abs=1e-14)
    np.testing.assert_array_equal(dc.p,d.p[:k]);np.testing.assert_array_equal(dc.m,d.m[:k])
    result=evaluate(d,h,k)
    assert result['TE_D_to_H']==pytest.approx(0,abs=1e-15)
    assert result['STD_DIFF']==pytest.approx(0,abs=1e-15)
    direct=(d.n[:k]*d.p[:k]*d.m[:k]).sum()/d.n[:k].sum()
    assert result['Mean_C_D_bp']/BP==pytest.approx(direct,abs=1e-15)
    assert result['shapley_residual']==pytest.approx(0,abs=1e-15)


def test_cap5_removes_tail_and_boundary():
    d=synthetic(np.full(10,100))
    m=d.m.copy();m[-1]=1000
    h=synthetic(np.full(10,100),m=m)
    assert evaluate(d,h,9)['TE_D_to_H']==pytest.approx(0,abs=1e-15)
    assert evaluate(d,h,10)['TE_D_to_H']>1
    assert assign_bins(np.array([1.5,2,3,5])).tolist()==[6,7,8,9]


def test_undefined_retained_invalid_excluded_ignored():
    p=np.full(10,.5);p[-1]=0
    d=synthetic(np.full(10,100),p=p)
    h=synthetic(np.full(10,100))
    assert np.isnan(d.m[-1])
    assert evaluate(d,h,10) is None
    assert evaluate(d,h,9) is not None
    p[1]=0
    assert evaluate(synthetic(np.full(10,100),p=p),h,6) is None


@pytest.mark.skipif(not DERIVED_FIXTURES.is_dir(), reason="checked-in support-ladder fixtures missing")
@pytest.mark.parametrize('market',['ETH5','BASE5'])
def test_empirical_counts_means_and_all_replicates(market):
    maps=[]
    for window in ['discovery','confirmation']:
        a=np.load(DERIVED_FIXTURES/f'{market}_{window}_daily.npz')
        m=aggregate_day_stats(a['n'],a['ns'],a['sc']);maps.append(m)
        counts=[]
        for cap,k in CAPS:
            c=capped_map(m,k);counts.append(c.n.sum())
            assert c.pi.sum()==pytest.approx(1,abs=1e-14)
            direct=a['sc'][:,:k].sum()/a['n'][:,:k].sum()
            assert (c.pi*c.p*c.m).sum()==pytest.approx(direct,abs=1e-12)
        assert np.all(np.diff(counts)>=0)
        assert counts[-2]==counts[-1]-m.n[-1]
    for cap,k in CAPS:
        assert abs(evaluate(*maps,k)['shapley_residual'])<1e-12
    a=np.load(DERIVED_FIXTURES/f'{market}_bootstrap.npz')
    assert a['draws_bp'].shape==(2000,5,5)
    assert np.nanmax(np.abs(a['shapley_residual_native']))<1e-12
    assert np.nanmax(np.abs(a['mean_residual_native']))<1e-12
    for j,(cap,k) in enumerate(CAPS):
        defined=np.isfinite(a['draws_bp'][:,j,:]).all(axis=1)
        np.testing.assert_array_equal(np.isfinite(a['shapley_residual_native'][:,j]),defined)


@pytest.mark.skipif(not FIXTURES.joinpath("frozen_rmt_support_full_table.csv").is_file(), reason="checked-in support-ladder fixtures missing")
def test_full_reproduces_frozen():
    old={r['Market']:r for r in csv.DictReader((RMT/'RESPONSE_MAP_TRANSPORT_TABLE.csv').open())}
    rows=list(csv.DictReader((FIXTURES/'frozen_rmt_support_full_table.csv').open()))
    assert len(rows)==10
    for r in rows:
        if r['Support_cap']=='Full': assert check_full(r,old[r['Market']])<=1e-10
