"""Algorithmic smoke tests for pycoloc — no R required.

These check the internal consistency of each ported routine on synthetic
GWAS summary statistics and on the bundled ``coloc_test_data`` (exported
to TSV by the R driver, when available).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import pycoloc as cl

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_REF = HERE / "r_ref_out"
_HAS_REF = (R_REF / "D1.tsv").exists()


# ----------------------------------------------------------------------
# synthetic fixtures
# ----------------------------------------------------------------------
def _make_dataset(rng, n=200, causal=50, effect=0.4, type="quant",
                  with_ld=False):
    """A small synthetic single-trait dataset with one causal SNP."""
    snp = [f"s{i + 1}" for i in range(n)]
    position = np.arange(1, n + 1) * 1000
    maf = rng.uniform(0.05, 0.5, n)
    se = rng.uniform(0.03, 0.06, n)
    beta = rng.normal(0, 0.02, n)
    beta[causal] = effect
    varbeta = se ** 2
    d = dict(snp=snp, position=position, beta=beta, varbeta=varbeta,
             MAF=maf, type=type, N=2000)
    if type == "quant":
        d["sdY"] = 1.0
    else:
        d["s"] = 0.5
    if with_ld:
        ld = np.eye(n)
        d["LD"] = pd.DataFrame(ld, index=snp, columns=snp)
    return d


@pytest.fixture(scope="module")
def synthetic_pair():
    rng = np.random.default_rng(0)
    d1 = _make_dataset(rng, causal=50, with_ld=True)
    rng2 = np.random.default_rng(1)
    d2 = _make_dataset(rng2, causal=50, with_ld=True)
    # share the causal SNP -> expect high H4
    return d1, d2


@pytest.fixture(scope="module")
def test_data():
    """Bundled coloc_test_data D1/D2, if the R driver has exported it."""
    if not _HAS_REF:
        pytest.skip("R reference data not exported.")
    out = {}
    meta = pd.read_csv(R_REF / "D_meta.tsv", sep="\t").set_index("trait")
    for p in ("D1", "D2"):
        df = pd.read_csv(R_REF / f"{p}.tsv", sep="\t")
        ld = pd.read_csv(R_REF / f"{p}_LD.tsv", sep="\t", index_col=0)
        out[p] = dict(
            snp=df["snp"].tolist(), position=df["position"].values,
            beta=df["beta"].values, varbeta=df["varbeta"].values,
            MAF=df["MAF"].values, type=meta.loc[p, "type"],
            N=int(meta.loc[p, "N"]), sdY=float(meta.loc[p, "sdY"]),
            LD=ld,
        )
    return out


# ----------------------------------------------------------------------
# utility helpers
# ----------------------------------------------------------------------
def test_logsum_logdiff():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(cl.logsum(x), np.log(np.sum(np.exp(x))))
    assert np.isclose(cl.logdiff(3.0, 1.0),
                      np.log(np.exp(3.0) - np.exp(1.0)))


def test_var_data():
    assert np.isclose(cl.Var_data(0.3, 1000), 1 / (2 * 1000 * 0.3 * 0.7))
    assert np.isclose(cl.Var_data_cc(0.3, 1000, 0.5),
                      1 / (2 * 1000 * 0.3 * 0.7 * 0.5 * 0.5))


def test_sdY_est_recovers_known_sdY():
    rng = np.random.default_rng(2)
    n = 500
    maf = rng.uniform(0.05, 0.5, n)
    N = 3000
    sdY_true = 2.0
    varbeta = sdY_true ** 2 / (2 * N * maf * (1 - maf))
    est = cl.sdY_est(varbeta, maf, N)
    assert np.isclose(est, sdY_true, rtol=1e-6)


def test_adjust_prior_caps():
    assert cl.adjust_prior(1e-4, 100) == 1e-4
    assert cl.adjust_prior(0.1, 100) == pytest.approx(1 / 101)


# ----------------------------------------------------------------------
# dataset checks
# ----------------------------------------------------------------------
def test_check_dataset_valid(synthetic_pair):
    d1, _ = synthetic_pair
    assert cl.check_dataset(d1, "1") is None


def test_check_dataset_rejects_missing_type():
    with pytest.raises(ValueError):
        cl.check_dataset({"snp": ["a", "b"]}, "x")


def test_check_dataset_rejects_bad_maf():
    rng = np.random.default_rng(3)
    d = _make_dataset(rng, n=60, causal=10)
    d["MAF"] = np.full(60, 1.5)
    with pytest.raises(ValueError):
        cl.check_dataset(d)


def test_process_dataset_estimates_path(synthetic_pair):
    d1, _ = synthetic_pair
    df = cl.process_dataset(d1, "df1")
    assert "lABF.df1" in df.columns
    assert len(df) == len(d1["snp"])


def test_process_dataset_pvalue_path():
    rng = np.random.default_rng(4)
    d = _make_dataset(rng, n=80)
    from scipy.stats import norm

    p = norm.cdf(-np.abs(d["beta"] / np.sqrt(d["varbeta"]))) * 2
    p = np.clip(p, 1e-300, 1.0)
    dp = dict(snp=d["snp"], pvalues=p, MAF=d["MAF"], N=d["N"], type="quant")
    df = cl.process_dataset(dp, "")
    assert "lABF." in df.columns


def test_subset_dataset(synthetic_pair):
    d1, _ = synthetic_pair
    sub = cl.subset_dataset(d1, np.arange(10))
    assert len(sub["snp"]) == 10
    assert sub["LD"].shape == (10, 10)


# ----------------------------------------------------------------------
# coloc.abf / finemap.abf
# ----------------------------------------------------------------------
def test_coloc_abf_shared_causal_high_h4(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_abf(d1, d2)
    assert set(["nsnps"] + [f"PP.H{i}.abf" for i in range(5)]) <= set(
        res["summary"].index
    )
    pp = res["summary"][[f"PP.H{i}.abf" for i in range(5)]].values
    assert np.isclose(pp.sum(), 1.0)
    assert res["summary"]["PP.H4.abf"] > 0.5


def test_coloc_abf_distinct_causal_high_h3():
    rng = np.random.default_rng(10)
    d1 = _make_dataset(rng, causal=20, effect=0.5)
    d2 = _make_dataset(np.random.default_rng(11), causal=120, effect=0.5)
    res = cl.coloc_abf(d1, d2)
    assert res["summary"]["PP.H3.abf"] > res["summary"]["PP.H4.abf"]


def test_finemap_abf_posterior_sums_to_one(synthetic_pair):
    d1, _ = synthetic_pair
    fm = cl.finemap_abf(d1)
    assert "SNP.PP" in fm.columns
    assert (fm["snp"] == "null").any()
    assert np.isclose(fm["SNP.PP"].sum(), 1.0)


def test_finemap_abf_picks_causal(synthetic_pair):
    d1, _ = synthetic_pair
    fm = cl.finemap_abf(d1).set_index("snp")
    best = fm.drop("null")["SNP.PP"].idxmax()
    assert best == "s51"  # 0-based causal index 50


def test_combine_abf_normalised():
    rng = np.random.default_rng(5)
    l1 = rng.normal(0, 2, 50)
    l2 = rng.normal(0, 2, 50)
    pp = cl.combine_abf(l1, l2, 1e-4, 1e-4, 1e-5)
    assert np.isclose(pp.sum(), 1.0)


def test_coloc_detail_has_h3_grid(synthetic_pair):
    d1, d2 = synthetic_pair
    det = cl.coloc_detail(d1, d2)
    assert "results.H3" in det
    n = len(det["results"])
    assert len(det["results.H3"]) == n * n


# ----------------------------------------------------------------------
# susie-based coloc
# ----------------------------------------------------------------------
def test_logbf_to_pp():
    bf = np.array([[2.0, 1.0, 0.0], [0.5, 3.0, 0.0]])
    pp = cl.logbf_to_pp(bf, 1e-4, last_is_null=True)
    assert pp.shape == (2, 3)
    assert np.allclose(pp.sum(axis=1), 1.0)


def test_coloc_bf_bf_single_pair():
    rng = np.random.default_rng(6)
    p = 60
    bf = rng.normal(0, 1, p)
    bf[30] = 10.0  # strong shared signal
    snps = [f"s{i}" for i in range(p)]
    res = cl.coloc_bf_bf(bf, bf, snps1=snps, snps2=snps)
    assert res["summary"]["PP.H4.abf"].iloc[0] > 0.9


def test_coloc_susie_accepts_matrix():
    rng = np.random.default_rng(7)
    p = 80
    lbf = rng.normal(0, 1, (3, p))
    lbf[0, 40] = 12.0
    lbf2 = rng.normal(0, 1, (2, p))
    lbf2[0, 40] = 12.0
    snps = [f"s{i}" for i in range(p)]
    res = cl.coloc_susie(lbf, lbf2, snps1=snps, snps2=snps,
                         cs_index1=[1], cs_index2=[1])
    assert "PP.H4.abf" in res["summary"].columns
    assert res["summary"]["PP.H4.abf"].iloc[0] > 0.9


def test_coloc_susie_accepts_dict():
    rng = np.random.default_rng(8)
    p = 50
    snps = [f"s{i}" for i in range(p)]
    lbf = rng.normal(0, 1, (2, p))
    lbf[0, 25] = 11.0
    s1 = {"lbf_variable": lbf, "snp": snps,
          "sets": {"cs_index": [1]}}
    res = cl.coloc_susie(s1, s1)
    assert res["summary"]["PP.H4.abf"].iloc[0] > 0.9


def test_finemap_susie():
    rng = np.random.default_rng(9)
    p = 40
    lbf = rng.normal(0, 1, (1, p))
    lbf[0, 10] = 9.0
    snps = [f"s{i}" for i in range(p)]
    res = cl.finemap_susie(lbf, snps=snps, cs_index=[1])
    assert "idx" in res["summary"].columns


# ----------------------------------------------------------------------
# signals (conditioning / masking)
# ----------------------------------------------------------------------
def test_coloc_signals_single(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_signals(d1, d2, method="single", p12=1e-5)
    assert "PP.H4.abf" in res["summary"].columns
    assert res["summary"]["PP.H4.abf"].iloc[0] > 0.5


def test_coloc_signals_cond_runs(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_signals(d1, d2, method="cond", p12=1e-5)
    assert len(res["summary"]) >= 1
    pp = res["summary"][[f"PP.H{i}.abf" for i in range(5)]]
    assert np.allclose(pp.sum(axis=1), 1.0)


def test_coloc_signals_mask_runs(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_signals(d1, d2, method="mask", p12=1e-5)
    assert "hit1" in res["summary"].columns


def test_coloc_signals_requires_p12(synthetic_pair):
    d1, d2 = synthetic_pair
    with pytest.raises(ValueError):
        cl.coloc_signals(d1, d2, method="single")


def test_finemap_signals(synthetic_pair):
    d1, _ = synthetic_pair
    hits = cl.finemap_signals(d1, method="cond", maxhits=3)
    assert isinstance(hits, dict)
    assert "s51" in hits


# ----------------------------------------------------------------------
# bin2lin (case/control linear approximation)
# ----------------------------------------------------------------------
def test_bin2lin_quality_near_one():
    rng = np.random.default_rng(12)
    n = 100
    d = _make_dataset(rng, n=n, type="cc")
    d["beta"] = rng.normal(0, 0.05, n)
    d["varbeta"] = np.full(n, 0.01)
    lin = cl.bin2lin(d)
    assert 0.5 < lin["quality"] < 1.5


# ----------------------------------------------------------------------
# sensitivity & plots
# ----------------------------------------------------------------------
def test_sensitivity_returns_grid(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_abf(d1, d2)
    sens = cl.sensitivity(res, "H4 > 0.5", npoints=50, doplot=False)
    assert len(sens) == 50
    assert "pass" in sens.columns
    assert set("H0 H1 H2 H3 H4".split()) <= set(sens.columns)


def test_sensitivity_plot(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_abf(d1, d2)
    cl.sensitivity(res, "H4 > 0.8", dataset1=d1, dataset2=d2,
                   npoints=30, doplot=True)
    import matplotlib.pyplot as plt

    assert plt.get_fignums()
    plt.close("all")


def test_plot_coloc_abf(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_abf(d1, d2)
    fig = cl.plot_coloc_abf(res)
    assert isinstance(fig, Figure)


def test_plot_dataset(synthetic_pair):
    d1, _ = synthetic_pair
    ax = cl.plot_dataset(d1)
    assert ax is not None


def test_manhattan(synthetic_pair):
    d1, d2 = synthetic_pair
    res = cl.coloc_abf(d1, d2)
    ax = cl.manhattan(res)
    assert ax is not None


def test_check_alignment(synthetic_pair):
    d1, _ = synthetic_pair
    frac = cl.check_alignment(d1, do_plot=False)
    assert 0.0 <= frac <= 1.0


# ----------------------------------------------------------------------
# prior helpers
# ----------------------------------------------------------------------
def test_prior_snp2hyp_sums_to_one():
    hp = cl.prior_snp2hyp(500, p12=1e-5, p1=1e-4, p2=1e-4)
    assert np.isclose(hp.sum(axis=1).iloc[0], 1.0)


def test_prior_adjust_renormalises():
    summ = pd.Series({
        "nsnps": 500, "PP.H0.abf": 0.1, "PP.H1.abf": 0.1,
        "PP.H2.abf": 0.1, "PP.H3.abf": 0.2, "PP.H4.abf": 0.5,
    })
    adj = cl.prior_adjust(summ, newp12=np.array([1e-6, 1e-5]),
                          p1=1e-4, p2=1e-4, p12=1e-5)
    assert np.allclose(adj.sum(axis=1), 1.0)


# ----------------------------------------------------------------------
# bundled test data (no R needed once exported)
# ----------------------------------------------------------------------
def test_bundled_coloc_abf(test_data):
    res = cl.coloc_abf(test_data["D1"], test_data["D2"])
    assert res["summary"]["PP.H4.abf"] > 0.99
