"""Single-causal-variant colocalisation via Approximate Bayes Factors.

Ports the workhorse functions of coloc:

* :func:`coloc_abf`     — ``coloc:::coloc.abf``
* :func:`coloc_detail`  — ``coloc:::coloc.detail`` (also returns the H3 grid)
* :func:`finemap_abf`   — ``coloc:::finemap.abf``
* :func:`combine_abf`   — ``coloc:::combine.abf``
* :func:`prior_snp2hyp` — ``coloc:::prior.snp2hyp``
* :func:`prior_adjust`  — ``coloc:::prior.adjust``

The colocalisation framework follows Giambartolomei *et al.* (2014) and the
single-variant ABF of Wakefield (2009).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import check_dataset, process_dataset
from .utils import adjust_prior, logdiff, logsum

__all__ = [
    "coloc_abf",
    "coloc_detail",
    "finemap_abf",
    "combine_abf",
    "prior_snp2hyp",
    "prior_adjust",
    "ColocABF",
]

_PP_NAMES = ["PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf"]
_SUMMARY_NAMES = ["nsnps"] + _PP_NAMES


class ColocABF(dict):
    """Result container for :func:`coloc_abf` (the R ``coloc_abf`` class).

    Behaves as a dict with keys ``summary``, ``results`` and ``priors``.
    A convenience :meth:`print` reproduces ``coloc:::print.coloc_abf``.
    """

    def __repr__(self):
        summ = self["summary"]
        if isinstance(summ, pd.DataFrame):
            return f"<ColocABF: {len(summ)} signal-pair(s)>"
        pp = " ".join(
            f"{k.replace('PP.', '').replace('.abf', '')}={v:.3g}"
            for k, v in summ.items() if k != "nsnps"
        )
        return f"<ColocABF: nsnps={int(summ['nsnps'])} {pp}>"

    def summary_str(self, trait1="trait 1", trait2="trait 2"):
        """Human-readable summary, mirroring ``print.coloc_abf``."""
        lines = [f"Coloc analysis of {trait1}, {trait2}", "", "SNP Priors"]
        lines.append(str(self["priors"]))
        summ = self["summary"]
        ns = (summ["nsnps"].iloc[0]
              if isinstance(summ, pd.DataFrame) else summ["nsnps"])
        pr = self["priors"]
        hp = prior_snp2hyp(ns, p1=pr["p1"], p2=pr["p2"], p12=pr["p12"])
        lines += ["", "Hypothesis Priors", str(hp), "", "Posterior",
                  str(summ)]
        return "\n".join(lines)


def combine_abf(l1, l2, p1, p2, p12, quiet=True):
    """Combine per-SNP log-ABFs of two traits into PP.H0..H4.

    Faithful port of ``coloc:::combine.abf``.

    Parameters
    ----------
    l1, l2 : array_like
        Per-SNP log approximate Bayes factors for trait 1 and trait 2.
    p1, p2, p12 : float
        Prior probabilities a SNP is causal for trait 1 only, trait 2
        only, or both.
    quiet : bool
        If ``False`` print the posterior probabilities.

    Returns
    -------
    pandas.Series
        ``PP.H0.abf`` ... ``PP.H4.abf``.
    """
    l1 = np.asarray(l1, dtype=float)
    l2 = np.asarray(l2, dtype=float)
    if len(l1) != len(l2):
        raise ValueError("l1 and l2 must have equal length")
    lsum = l1 + l2
    lH0 = 0.0
    lH1 = np.log(p1) + logsum(l1)
    lH2 = np.log(p2) + logsum(l2)
    lH3 = (np.log(p1) + np.log(p2)
           + logdiff(logsum(l1) + logsum(l2), logsum(lsum)))
    lH4 = np.log(p12) + logsum(lsum)
    all_abf = np.array([lH0, lH1, lH2, lH3, lH4])
    denom = logsum(all_abf)
    pp = np.exp(all_abf - denom)
    out = pd.Series(pp, index=_PP_NAMES)
    if not quiet:
        print(out.round(3))
        print(f"PP abf for shared variant: "
              f"{round(out['PP.H4.abf'], 3) * 100}%")
    return out


def coloc_abf(dataset1, dataset2, MAF=None, p1=1e-4, p2=1e-4, p12=1e-5):
    """Bayesian colocalisation of two traits under a single causal variant.

    Faithful port of ``coloc::coloc.abf`` (Giambartolomei *et al.* 2014).
    Each dataset is a :class:`dict` of GWAS summary statistics for one
    trait; see :mod:`pycoloc.dataset` for the recognised keys.

    Parameters
    ----------
    dataset1, dataset2 : dict
        Datasets for the two traits.
    MAF : array_like, optional
        Minor allele frequencies, attached to datasets lacking ``MAF``.
    p1, p2, p12 : float
        Prior probabilities a SNP is associated with trait 1, trait 2, or
        both.

    Returns
    -------
    ColocABF
        Keys ``summary`` (``nsnps`` and ``PP.H0..H4.abf``), ``results``
        (per-SNP table with ``lABF.df1/df2``, ``internal.sum.lABF`` and
        ``SNP.PP.H4``) and ``priors``.
    """
    if "MAF" not in dataset1 and MAF is not None:
        dataset1 = {**dataset1, "MAF": MAF}
    if "MAF" not in dataset2 and MAF is not None:
        dataset2 = {**dataset2, "MAF": MAF}
    check_dataset(d=dataset1, suffix="1")
    check_dataset(d=dataset2, suffix="2")
    df1 = process_dataset(d=dataset1, suffix="df1")
    df2 = process_dataset(d=dataset2, suffix="df2")
    p1 = adjust_prior(p1, len(df1), "1")
    p2 = adjust_prior(p2, len(df2), "2")
    merged = _merge_dfs(df1, df2)
    p12 = adjust_prior(p12, len(merged), "12")
    if len(merged) == 0:
        raise ValueError(
            "dataset1 and dataset2 should contain the same snps in the "
            "same order, or share snp names."
        )
    merged["internal.sum.lABF"] = merged["lABF.df1"] + merged["lABF.df2"]
    denom = logsum(merged["internal.sum.lABF"].values)
    merged["SNP.PP.H4"] = np.exp(merged["internal.sum.lABF"].values - denom)
    pp = combine_abf(merged["lABF.df1"].values, merged["lABF.df2"].values,
                     p1, p2, p12, quiet=True)
    summary = pd.Series(
        [len(merged)] + list(pp.values), index=_SUMMARY_NAMES
    )
    out = ColocABF(
        summary=summary,
        results=merged,
        priors=pd.Series({"p1": p1, "p2": p2, "p12": p12}),
    )
    return out


def coloc_detail(dataset1, dataset2, MAF=None, p1=1e-4, p2=1e-4, p12=1e-5):
    """Colocalisation with the full per-SNP-pair H3 grid.

    Port of ``coloc:::coloc.detail``.  Like :func:`coloc_abf` but also
    returns ``results.H3``, the all-pairs ``snp1 x snp2`` table of
    ``lABF.h3`` used by :func:`pycoloc.coloc_signals`.
    """
    if not isinstance(dataset1, dict) or not isinstance(dataset2, dict):
        raise ValueError("dataset1 and dataset2 must be dicts.")
    if "MAF" not in dataset1 and MAF is not None:
        dataset1 = {**dataset1, "MAF": MAF}
    if "MAF" not in dataset2 and MAF is not None:
        dataset2 = {**dataset2, "MAF": MAF}
    df1 = process_dataset(d=dataset1, suffix="df1")
    df1 = df1[~df1["lABF.df1"].isna()].reset_index(drop=True)
    df2 = process_dataset(d=dataset2, suffix="df2")
    df2 = df2[~df2["lABF.df2"].isna()].reset_index(drop=True)
    df = _merge_dfs(df1, df2)
    if len(df) == 0:
        raise ValueError("datasets share no snps")
    df["lABF.h4"] = df["lABF.df1"] + df["lABF.df2"]
    denom = logsum(df["lABF.h4"].values)
    df["SNP.PP.H4"] = np.exp(df["lABF.h4"].values - denom)
    # all-pairs grid for H3
    s1 = df1[["snp", "lABF.df1"]]
    s2 = df2[["snp", "lABF.df2"]]
    grid = pd.MultiIndex.from_product(
        [s1["snp"].values, s2["snp"].values], names=["snp1", "snp2"]
    ).to_frame(index=False)
    grid = grid.merge(s1.rename(columns={"snp": "snp1"}), on="snp1")
    grid = grid.merge(s2.rename(columns={"snp": "snp2"}), on="snp2")
    grid["lABF.h3"] = grid["lABF.df1"] + grid["lABF.df2"]
    df = df.rename(columns={"lABF.df1": "lABF.h1", "lABF.df2": "lABF.h2"})
    pp = combine_abf(df["lABF.h1"].values, df["lABF.h2"].values,
                     p1, p2, p12, quiet=True)
    summary = pd.Series([len(df)] + list(pp.values), index=_SUMMARY_NAMES)
    return {
        "summary": summary,
        "results": df,
        "results.H3": grid,
        "priors": pd.Series({"p1": p1, "p2": p2, "p12": p12}),
    }


def finemap_abf(dataset, p1=1e-4):
    """Single-trait ABF fine-mapping (per-SNP posterior probabilities).

    Faithful port of ``coloc::finemap.abf``.

    Parameters
    ----------
    dataset : dict
        A coloc dataset.
    p1 : float
        Prior probability a SNP is causal.

    Returns
    -------
    pandas.DataFrame
        The per-SNP table with a ``null`` row appended, plus ``prior``
        and ``SNP.PP`` columns (the posterior probability per SNP).
    """
    check_dataset(dataset, "")
    df = process_dataset(d=dataset, suffix="")
    nsnps = len(df)
    p1 = adjust_prior(p1, nsnps, "1")
    # append a null row
    null = {c: np.nan for c in df.columns}
    null["snp"] = "null"
    null["lABF."] = 0.0
    df = pd.concat([df, pd.DataFrame([null])], ignore_index=True)
    df["prior"] = list(np.repeat(p1, nsnps)) + [1 - nsnps * p1]
    lab = df["lABF."].values
    log_prior = np.log(df["prior"].values)
    denom = logsum(lab + log_prior)
    df["SNP.PP"] = np.exp(lab + log_prior - denom)
    return df


def prior_snp2hyp(nsnp, p12=1e-6, p1=1e-4, p2=1e-4):
    """Convert per-SNP priors into hypothesis priors H0..H4.

    Port of ``coloc:::prior.snp2hyp``.  Returns ``None`` if the priors are
    incoherent.

    Parameters
    ----------
    nsnp : int or array_like
        Number of SNPs.
    p12, p1, p2 : float or array_like
        Per-SNP priors.

    Returns
    -------
    pandas.DataFrame or None
        Columns ``H0`` ... ``H4``.
    """
    p12 = np.atleast_1d(np.asarray(p12, dtype=float))
    nsnp = np.asarray(nsnp, dtype=float)
    if (np.any(p12 < p1 * p2) or np.any(p12 > p1) or np.any(p12 > p2)):
        return None
    h1 = nsnp * p1
    h2 = nsnp * p2
    h3 = nsnp * (nsnp - 1) * p1 * p2
    h4 = nsnp * p12
    h1, h2, h3, h4 = np.broadcast_arrays(h1, h2, h3, h4)
    h0 = 1 - (h1 + h2 + h3 + h4)
    tmp = np.column_stack([h0, h1, h2, h3, h4])
    return pd.DataFrame(tmp, columns=[f"H{i}" for i in range(5)])


def prior_adjust(summ, newp12, p1=1e-4, p2=1e-4, p12=1e-6):
    """Re-weight a coloc summary under a new ``p12`` prior.

    Port of ``coloc:::prior.adjust`` — the engine behind
    :func:`pycoloc.sensitivity`.

    Parameters
    ----------
    summ : pandas.Series or dict
        A coloc summary vector (``nsnps``, ``PP.H0..H4.abf``).
    newp12 : float or array_like
        The new ``p12`` value(s) to evaluate.
    p1, p2, p12 : float
        The original priors.

    Returns
    -------
    pandas.DataFrame
        Re-normalised posterior probabilities, one row per ``newp12``.
    """
    if isinstance(summ, (dict, ColocABF)) and "summary" in summ:
        summ = summ["summary"]
    nsnp = summ["nsnps"]
    pr1 = prior_snp2hyp(nsnp, p12=newp12, p1=p1, p2=p2)
    pr0_single = prior_snp2hyp(nsnp, p12=p12, p1=p1, p2=p2)
    n = len(pr1)
    pr0 = pd.concat([pr0_single] * n, ignore_index=True)
    pp = np.array([summ[k] for k in _PP_NAMES], dtype=float)
    pp_mat = np.tile(pp, (n, 1))
    newpp = pp_mat * pr1.values / pr0.values
    newpp = newpp / newpp.sum(axis=1, keepdims=True)
    return pd.DataFrame(newpp, columns=[f"H{i}" for i in range(5)])


def _merge_dfs(df1, df2):
    """Inner-merge two processed datasets on shared columns (R ``merge``)."""
    common = [c for c in df1.columns if c in df2.columns]
    merged = df1.merge(df2, on=common, how="inner")
    return merged.reset_index(drop=True)
