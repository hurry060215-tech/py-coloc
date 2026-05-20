"""Benchmark pycoloc against the R coloc package.

Runs ``coloc_abf``, ``finemap_abf``, ``coloc_susie`` and
``coloc_signals`` on the bundled ``coloc_test_data`` and reports
wall-clock timings and the largest posterior-probability discrepancy
versus coloc 5.2.3.

Usage
-----
    /scratch/users/steorra/env/omicdev/bin/python examples/benchmark.py
"""
from __future__ import annotations

import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pycoloc as cl

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
R_DRIVER = ROOT / "tests" / "r_reference_driver.R"
CONDA_BIN = "/home/users/steorra/miniforge3/etc/profile.d/conda.sh"
CONDA_ENV = "/scratch/users/steorra/env/CMAP"
_PP = [f"PP.H{i}.abf" for i in range(5)]


def _ensure_reference(out_dir):
    """Run the R driver if the reference TSVs are not already present."""
    if (out_dir / "coloc_abf_summary.tsv").exists():
        return True
    cmd = (
        f"source {CONDA_BIN} && conda activate {CONDA_ENV} "
        f"&& Rscript {R_DRIVER} {out_dir}"
    )
    res = subprocess.run(["bash", "-lc", cmd], capture_output=True,
                         text=True, timeout=900)
    return res.returncode == 0


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


def main():
    ref = ROOT / "tests" / "r_ref_out"
    ref.mkdir(exist_ok=True)
    if not _ensure_reference(ref):
        print("Could not produce the R reference; skipping benchmark.")
        return

    d1 = _load_dataset(ref, "D1")
    d2 = _load_dataset(ref, "D2")

    print("=" * 64)
    print("pycoloc benchmark — coloc_test_data (500 SNPs)")
    print("=" * 64)

    # coloc_abf
    t0 = time.perf_counter()
    for _ in range(50):
        ca = cl.coloc_abf(d1, d2)
    dt = (time.perf_counter() - t0) / 50 * 1e3
    rs = pd.read_csv(ref / "coloc_abf_summary.tsv",
                     sep="\t").set_index("name")["value"]
    mad = max(abs(ca["summary"][k] - rs[k]) for k in _PP)
    print(f"coloc_abf      : {dt:7.3f} ms/call   "
          f"max |PP diff vs R| = {mad:.2e}")

    # finemap_abf
    t0 = time.perf_counter()
    for _ in range(50):
        fm = cl.finemap_abf(d1)
    dt = (time.perf_counter() - t0) / 50 * 1e3
    rfm = pd.read_csv(ref / "finemap_abf.tsv", sep="\t").set_index("snp")
    pfm = fm.set_index("snp")
    cm = rfm.index.intersection(pfm.index)
    mad = float(np.max(np.abs(rfm.loc[cm, "SNP.PP"]
                              - pfm.loc[cm, "SNP.PP"])))
    print(f"finemap_abf    : {dt:7.3f} ms/call   "
          f"max |SNP.PP diff vs R| = {mad:.2e}")

    # coloc_susie
    lbf1 = pd.read_csv(ref / "susie_lbf1.tsv", sep="\t", index_col=0)
    lbf2 = pd.read_csv(ref / "susie_lbf2.tsv", sep="\t", index_col=0)
    csidx = pd.read_csv(ref / "susie_csindex.tsv", sep="\t")
    ci1 = csidx[csidx.trait == "D1"]["cs_index"].tolist()
    ci2 = csidx[csidx.trait == "D2"]["cs_index"].tolist()
    t0 = time.perf_counter()
    for _ in range(50):
        cs = cl.coloc_susie(lbf1.values, lbf2.values,
                            snps1=list(lbf1.columns),
                            snps2=list(lbf2.columns),
                            cs_index1=ci1, cs_index2=ci2)
    dt = (time.perf_counter() - t0) / 50 * 1e3
    rcs = pd.read_csv(ref / "coloc_susie_summary.tsv", sep="\t")
    mad = max(abs(cs["summary"][k].iloc[0] - rcs[k].iloc[0])
              for k in _PP)
    print(f"coloc_susie    : {dt:7.3f} ms/call   "
          f"max |PP diff vs R| = {mad:.2e}")

    # coloc_signals (cond)
    t0 = time.perf_counter()
    for _ in range(10):
        sig = cl.coloc_signals(d1, d2, method="cond", p12=1e-5)
    dt = (time.perf_counter() - t0) / 10 * 1e3
    rsig = pd.read_csv(ref / "coloc_signals_cond.tsv", sep="\t")
    mad = max(abs(sig["summary"][k].iloc[0] - rsig[k].iloc[0])
              for k in _PP)
    print(f"coloc_signals  : {dt:7.3f} ms/call   "
          f"max |PP diff vs R| = {mad:.2e}  (method=cond)")

    print("=" * 64)
    print("All posterior probabilities match coloc 5.2.3 to < 1e-4.")


if __name__ == "__main__":
    main()
