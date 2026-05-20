"""R-parity tests — pycoloc vs the R/CRAN package coloc 5.2.3.

The R driver (:file:`r_reference_driver.R`) runs coloc on its bundled
``coloc_test_data`` (datasets ``D1``/``D2``) and exports both the raw
inputs and the reference results to TSV.  The Python side then runs
:mod:`pycoloc` on the *identical* exported inputs, so both implementations
analyse exactly the same data.  We compare:

* ``coloc_abf``    — PP.H0..H4 agree to < 1e-4 absolute; per-SNP
  ``SNP.PP.H4`` Pearson r > 0.9999.
* ``finemap_abf``  — per-SNP ``SNP.PP`` Pearson r > 0.9999 (quant + cc).
* ``coloc_susie``  — per-credible-set-pair PP.H0..H4 agree to < 1e-4
  (fed the identical ``runsusie`` ``lbf_variable`` matrices).
* ``coloc_signals`` (cond / mask) — PP.H0..H4 agree to < 1e-4.
* ``sensitivity``  — the swept posterior grid agrees to < 1e-6.

Tests skip gracefully when the CMAP R env / coloc is unavailable.
"""
from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

import pycoloc as cl

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
R_DRIVER = HERE / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"

_PP = [f"PP.H{i}.abf" for i in range(5)]


def _r_available() -> bool:
    if not R_DRIVER.exists():
        return False
    try:
        out = subprocess.run(
            ["bash", "-lc",
             f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
             "&& Rscript -e 'library(coloc); cat(\"OK\")'"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        return out.returncode == 0 and "OK" in out.stdout
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _r_available(),
    reason="CMAP R env or coloc not installed.",
)


@pytest.fixture(scope="module")
def r_reference(tmp_path_factory):
    """Run the coloc R reference once; return the output directory."""
    out_dir = tmp_path_factory.mktemp("coloc_R")
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    res = subprocess.run(
        ["bash", "-lc", cmd], capture_output=True, text=True, timeout=900,
    )
    if res.returncode != 0:
        pytest.skip(f"R reference driver failed:\n{res.stderr[-2000:]}")
    return out_dir


def _load_dataset(ref, prefix):
    df = pd.read_csv(ref / f"{prefix}.tsv", sep="\t")
    ld = pd.read_csv(ref / f"{prefix}_LD.tsv", sep="\t", index_col=0)
    meta = pd.read_csv(ref / "D_meta.tsv", sep="\t").set_index("trait")
    return dict(
        snp=df["snp"].tolist(), position=df["position"].values,
        beta=df["beta"].values, varbeta=df["varbeta"].values,
        MAF=df["MAF"].values, type=meta.loc[prefix, "type"],
        N=int(meta.loc[prefix, "N"]), sdY=float(meta.loc[prefix, "sdY"]),
        LD=ld,
    )


@pytest.fixture(scope="module")
def datasets(r_reference):
    return (_load_dataset(r_reference, "D1"),
            _load_dataset(r_reference, "D2"))


# ----------------------------------------------------------------------
# coloc.abf
# ----------------------------------------------------------------------
def test_coloc_abf_summary_vs_R(r_reference, datasets):
    d1, d2 = datasets
    res = cl.coloc_abf(d1, d2)
    rs = pd.read_csv(r_reference / "coloc_abf_summary.tsv",
                     sep="\t").set_index("name")["value"]
    for k in _PP:
        diff = abs(res["summary"][k] - rs[k])
        assert diff < 1e-4, f"{k}: py vs R abs diff {diff:.3e}"
    assert int(res["summary"]["nsnps"]) == int(rs["nsnps"])


def test_coloc_abf_snp_pp_h4_vs_R(r_reference, datasets):
    d1, d2 = datasets
    res = cl.coloc_abf(d1, d2)
    rr = pd.read_csv(r_reference / "coloc_abf_results.tsv",
                     sep="\t").set_index("snp")
    pr = res["results"].set_index("snp")
    cm = rr.index.intersection(pr.index)
    assert len(cm) == 500
    rho, _ = pearsonr(rr.loc[cm, "SNP.PP.H4"], pr.loc[cm, "SNP.PP.H4"])
    assert rho > 0.9999, f"SNP.PP.H4 Pearson r = {rho:.6f}"
    # lABF agreement should be near bit-exact
    mad = np.max(np.abs(rr.loc[cm, "lABF.df1"] - pr.loc[cm, "lABF.df1"]))
    assert mad < 1e-6, f"lABF.df1 max abs diff = {mad:.3e}"


# ----------------------------------------------------------------------
# finemap.abf
# ----------------------------------------------------------------------
def test_finemap_abf_vs_R(r_reference, datasets):
    d1, _ = datasets
    fm = cl.finemap_abf(d1).set_index("snp")
    rfm = pd.read_csv(r_reference / "finemap_abf.tsv",
                      sep="\t").set_index("snp")
    cm = rfm.index.intersection(fm.index)
    rho, _ = pearsonr(rfm.loc[cm, "SNP.PP"], fm.loc[cm, "SNP.PP"])
    assert rho > 0.9999, f"finemap_abf SNP.PP Pearson r = {rho:.6f}"
    mad = np.max(np.abs(rfm.loc[cm, "SNP.PP"] - fm.loc[cm, "SNP.PP"]))
    assert mad < 1e-4, f"finemap_abf SNP.PP max abs diff = {mad:.3e}"


def test_finemap_abf_cc_vs_R(r_reference, datasets):
    """Case/control p-value path of finemap.abf vs R."""
    d1, _ = datasets
    from scipy.stats import norm

    p = norm.cdf(-np.abs(d1["beta"] / np.sqrt(d1["varbeta"]))) * 2
    dcc = dict(snp=d1["snp"], pvalues=p, MAF=d1["MAF"],
               N=1000, type="cc", s=0.5)
    fm = cl.finemap_abf(dcc).set_index("snp")
    rfm = pd.read_csv(r_reference / "finemap_abf_cc.tsv",
                      sep="\t").set_index("snp")
    cm = rfm.index.intersection(fm.index)
    rho, _ = pearsonr(rfm.loc[cm, "SNP.PP"], fm.loc[cm, "SNP.PP"])
    assert rho > 0.9999, f"cc finemap_abf SNP.PP Pearson r = {rho:.6f}"


# ----------------------------------------------------------------------
# coloc.susie (fed identical runsusie lbf_variable matrices)
# ----------------------------------------------------------------------
def test_coloc_susie_vs_R(r_reference):
    lbf1 = pd.read_csv(r_reference / "susie_lbf1.tsv",
                       sep="\t", index_col=0)
    lbf2 = pd.read_csv(r_reference / "susie_lbf2.tsv",
                       sep="\t", index_col=0)
    csidx = pd.read_csv(r_reference / "susie_csindex.tsv", sep="\t")
    ci1 = csidx[csidx.trait == "D1"]["cs_index"].tolist()
    ci2 = csidx[csidx.trait == "D2"]["cs_index"].tolist()
    res = cl.coloc_susie(
        lbf1.values, lbf2.values,
        snps1=list(lbf1.columns), snps2=list(lbf2.columns),
        cs_index1=ci1, cs_index2=ci2,
    )
    rcs = pd.read_csv(r_reference / "coloc_susie_summary.tsv", sep="\t")
    assert len(res["summary"]) == len(rcs)
    for k in _PP:
        for i in range(len(rcs)):
            diff = abs(res["summary"][k].iloc[i] - rcs[k].iloc[i])
            assert diff < 1e-4, f"coloc_susie {k} row {i}: diff {diff:.3e}"
    # credible-set indices and lead SNPs match
    assert list(res["summary"]["idx1"]) == list(rcs["idx1"])
    assert list(res["summary"]["hit1"]) == list(rcs["hit1"])


def test_coloc_susie_per_snp_pp_vs_R(r_reference):
    lbf1 = pd.read_csv(r_reference / "susie_lbf1.tsv",
                       sep="\t", index_col=0)
    lbf2 = pd.read_csv(r_reference / "susie_lbf2.tsv",
                       sep="\t", index_col=0)
    csidx = pd.read_csv(r_reference / "susie_csindex.tsv", sep="\t")
    ci1 = csidx[csidx.trait == "D1"]["cs_index"].tolist()
    ci2 = csidx[csidx.trait == "D2"]["cs_index"].tolist()
    res = cl.coloc_susie(
        lbf1.values, lbf2.values,
        snps1=list(lbf1.columns), snps2=list(lbf2.columns),
        cs_index1=ci1, cs_index2=ci2,
    )
    rres_file = r_reference / "coloc_susie_results.tsv"
    if not rres_file.exists():
        pytest.skip("R produced no coloc.susie per-SNP results.")
    rres = pd.read_csv(rres_file, sep="\t").set_index("snp")
    pres = res["results"].set_index("snp")
    cm = rres.index.intersection(pres.index)
    ppcol = [c for c in rres.columns if "SNP.PP.H4" in c][0]
    pcol = [c for c in pres.columns if "SNP.PP.H4" in c][0]
    rho, _ = pearsonr(rres.loc[cm, ppcol], pres.loc[cm, pcol])
    assert rho > 0.9999, f"coloc_susie per-SNP PP Pearson r = {rho:.6f}"


# ----------------------------------------------------------------------
# coloc.signals
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["cond", "mask"])
def test_coloc_signals_vs_R(r_reference, datasets, method):
    d1, d2 = datasets
    res = cl.coloc_signals(d1, d2, method=method, p12=1e-5)
    rc = pd.read_csv(r_reference / f"coloc_signals_{method}.tsv", sep="\t")
    assert len(res["summary"]) == len(rc)
    for k in _PP:
        for i in range(len(rc)):
            diff = abs(res["summary"][k].iloc[i] - rc[k].iloc[i])
            assert diff < 1e-4, (
                f"coloc_signals[{method}] {k} row {i}: diff {diff:.3e}"
            )
    assert list(res["summary"]["hit1"]) == list(rc["hit1"])


# ----------------------------------------------------------------------
# sensitivity
# ----------------------------------------------------------------------
def test_sensitivity_vs_R(r_reference, datasets):
    d1, d2 = datasets
    res = cl.coloc_abf(d1, d2)
    sens = cl.sensitivity(res, "H4 > 0.5", npoints=100, doplot=False)
    rs = pd.read_csv(r_reference / "sensitivity.tsv", sep="\t")
    assert len(sens) == len(rs)
    for i, hcol in enumerate(_PP):
        pc = sens[f"H{i}"].values
        rc = rs[hcol].values
        mad = np.max(np.abs(pc - rc))
        assert mad < 1e-6, f"sensitivity H{i} max abs diff = {mad:.3e}"
    # the boolean pass vector must match exactly
    assert np.array_equal(sens["pass"].values,
                          rs["pass"].values.astype(bool))
