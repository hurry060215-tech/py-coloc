"""Multiple-signal colocalisation via conditioning and masking.

Ports coloc's signal-aware colocalisation machinery:

* :func:`coloc_signals`   — ``coloc::coloc.signals``
* :func:`finemap_signals` — ``coloc:::finemap.signals``
* :func:`est_all_cond`    — ``coloc:::est_all_cond``
* :func:`est_cond`        — ``coloc:::est_cond``
* :func:`coloc_process`   — ``coloc:::coloc.process``
* helpers ``map_cond``, ``map_mask``, ``find_best_signal``,
  ``bin2lin``, ``VMAF_cc`` etc.

Two strategies are supported (``method=``):

* ``"cond"`` — iteratively *condition* the effect estimates on the
  previously found lead SNPs (requires ``beta``, ``varbeta``, ``MAF``,
  ``N`` and an ``LD`` matrix).
* ``"mask"`` — *mask out* SNPs in LD with previously found signals.
* ``"single"`` — a single-signal analysis (default; equivalent to
  :func:`pycoloc.coloc_abf`).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .abf import ColocABF, coloc_detail
from .dataset import check_dataset, check_ld
from .utils import logsum, sdY_est

__all__ = [
    "coloc_signals",
    "finemap_signals",
    "est_all_cond",
    "est_cond",
    "coloc_process",
    "find_best_signal",
    "map_cond",
    "map_mask",
    "bin2lin",
    "VMAF_cc",
]


# ----------------------------------------------------------------------
# binary -> linear approximation
# ----------------------------------------------------------------------
def _vestgeno_ctl(f):
    f = np.asarray(f, dtype=float)
    return np.column_stack([(1 - f) ** 2, 2 * f * (1 - f), f ** 2])


def _vestgeno_cse(G0, b):
    b = np.asarray(b, dtype=float)
    g0 = 1.0
    g1 = np.exp(b + np.log(G0[:, 1]) - np.log(G0[:, 0]))
    g2 = np.exp(2 * b + np.log(G0[:, 2]) - np.log(G0[:, 0]))
    out = np.column_stack([np.full_like(g1, g0), g1, g2])
    return out / out.sum(axis=1, keepdims=True)


def VMAF_cc(f0, b, N0, N1):
    """Genotype variance for a case/control trait.

    Port of ``coloc:::VMAF.cc``.
    """
    G0 = _vestgeno_ctl(f0) * N0
    G1 = _vestgeno_cse(G0, b) * N1
    G = G0 + G1
    tot = N0 + N1
    E2 = G @ np.array([0.0, 1.0, 4.0]) / tot
    E1 = G @ np.array([0.0, 1.0, 2.0]) / tot
    return (E2 - E1 ** 2) * tot / (tot - 1)


def _wmean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if w.ndim == 2:
        return (x * w).sum(axis=0) / w.sum(axis=0)
    return (x * w).sum(axis=0) / w.sum()


def bin2lin(D):
    """Approximate a case/control trait by a linear model.

    Port of ``coloc:::bin2lin``.  Returns a copy of ``D`` with ``beta`` /
    ``varbeta`` replaced by the linearised approximation and a
    ``quality`` field measuring the fit (ideal is 1).
    """
    if D["type"] != "cc":
        raise ValueError("type != cc")
    s = D["s"]
    N = np.asarray(D["N"], dtype=float)
    maf = np.asarray(D["MAF"], dtype=float)
    beta = np.asarray(D["beta"], dtype=float)
    g0 = _vestgeno_ctl(maf)
    g1 = _vestgeno_cse(g0, beta)
    coef2 = np.array([0.0, 1.0, 2.0])
    coef4 = np.array([0.0, 1.0, 4.0])
    ex0 = g0 @ coef2
    ex1 = g1 @ coef2
    # meta version: s and N may be scalar -> single study
    ex = np.outer(np.atleast_1d(1 - s), ex0) + np.outer(np.atleast_1d(s), ex1)
    vx0 = g0 @ coef4
    vx1 = g1 @ coef4
    vx = (np.outer(np.atleast_1d(1 - s), vx0)
          + np.outer(np.atleast_1d(s), vx1) - ex ** 2)
    vy = np.atleast_1d(s * (1 - s))
    vxy = np.outer(vy, (ex1 - ex0))
    D1 = dict(D)
    # varbeta has shape (nstudies, nsnp); meta-combine over studies (axis 0)
    varbeta = (vy[:, None] / vx - vxy ** 2 / vx ** 2) / (N - 1)
    D1["varbeta"] = 1.0 / (1.0 / varbeta).sum(axis=0)
    D1["beta"] = _wmean(vxy / vx, 1.0 / varbeta)
    z1 = D1["beta"] / np.sqrt(D1["varbeta"])
    z = beta / np.sqrt(np.asarray(D["varbeta"], dtype=float))
    quality = float(np.sum(z * z1) / np.sum(z * z))
    if quality > 1.1 or quality < 0.9:
        warnings.warn("linear approx quality is not good")
    D1["quality"] = quality
    return D1


# ----------------------------------------------------------------------
# conditioning
# ----------------------------------------------------------------------
def _ld_matrix(D, LD=None):
    """Return the LD matrix as a snp-indexed DataFrame."""
    if LD is None:
        LD = D["LD"]
    if isinstance(LD, pd.DataFrame):
        return LD.loc[D["snp"], D["snp"]]
    arr = np.asarray(LD, dtype=float)
    snp = list(D["snp"])
    return pd.DataFrame(arr, index=snp, columns=snp)


def est_cond(x, LD, YY, sigsnps, xtx=None):
    """Estimate effect sizes conditional on a set of lead SNPs.

    Faithful port of ``coloc:::est_cond`` (the meta-analysis variant).

    Parameters
    ----------
    x : dict
        A dataset with ``beta``, ``varbeta``, ``MAF``, ``N``, ``snp``.
    LD : pandas.DataFrame or ndarray
        SNP correlation matrix.
    YY : float
        Total phenotype sum-of-squares ``sum(N * sdY^2)`` (quant) or
        ``sum(N * s * (1 - s))`` (cc).
    sigsnps : sequence of str
        SNP ids to condition on.
    xtx : ndarray, optional
        Precomputed ``X'X`` matrix.

    Returns
    -------
    pandas.DataFrame
        Columns ``snp``, ``beta``, ``varbeta`` (conditioned estimates).
    """
    ld = _ld_matrix(x, LD)
    snp = list(x["snp"])
    ld_arr = ld.values
    sigsnps = list(sigsnps)
    nuse = np.array([snp.index(s) for s in sigsnps])
    maxld2 = np.max(ld_arr[nuse, :] ** 2, axis=0)
    use = ~(np.isin(snp, sigsnps) | (maxld2 > 0.8))
    check_dataset(x, req=("beta", "varbeta", "MAF"))
    beta = np.asarray(x["beta"], dtype=float)
    N = np.asarray(x["N"], dtype=float)
    maf = np.asarray(x["MAF"], dtype=float)
    if xtx is not None:
        XX = np.asarray(xtx, dtype=float)
    elif x["type"] == "quant":
        VX = 2 * maf * (1 - maf)
        XX = np.sum(N) * ld_arr * np.sqrt(np.outer(VX, VX))
    else:
        s = x["s"]
        VW = VMAF_cc(maf, beta, N0=np.sum((1 - s) * N), N1=np.sum(s * N))
        XX = np.sum(N) * ld_arr * np.sqrt(np.outer(VW, VW))
    Dvec = np.diag(XX)
    D1v = Dvec[nuse]
    D2v = Dvec[use]
    XX1 = XX[np.ix_(nuse, nuse)]
    XX12 = XX[np.ix_(nuse, np.where(use)[0])]
    XX21 = XX[np.ix_(np.where(use)[0], nuse)]
    D1 = np.diag(np.atleast_1d(D1v))
    D2 = np.diag(np.atleast_1d(D2v))
    XX1inv = np.linalg.inv(XX1)
    b_nuse = beta[nuse].reshape(-1, 1)
    b1 = XX1inv @ D1 @ b_nuse
    diagD2 = np.diag(D2)
    b2 = (beta[use]
          - (XX21 @ XX1inv @ D1 @ b_nuse).ravel() / diagD2)
    Sc = float(YY - (b1.reshape(1, -1) @ D1 @ b_nuse))
    Sc = (Sc - b2 * diagD2 * beta[use]) / (np.sum(N) - len(sigsnps) - 1)
    vb2 = (Sc * (diagD2 - np.diag(XX21 @ XX1inv @ XX12)) / diagD2 ** 2)
    vb2 = np.abs(vb2)
    keep_snp = list(np.array(snp)[use])
    drop_snp = list(np.array(snp)[~use])
    part1 = pd.DataFrame({"snp": keep_snp, "beta": b2, "varbeta": vb2})
    part2 = pd.DataFrame({
        "snp": drop_snp, "beta": np.zeros(len(drop_snp)),
        "varbeta": np.asarray(x["varbeta"], dtype=float)[~use],
    })
    return pd.concat([part1, part2], ignore_index=True)


def est_all_cond(D, FM, mode="iterative"):
    """Build all conditioned datasets for a set of lead SNPs.

    Faithful port of ``coloc:::est_all_cond``.

    Parameters
    ----------
    D : dict
        The dataset.
    FM : dict
        Mapping of lead-SNP id -> marginal Z (the ``finemap.signals``
        output).
    mode : {"iterative", "allbutone"}
        Conditioning scheme.

    Returns
    -------
    dict
        Mapping conditioning-label -> conditioned dataset DataFrame.
    """
    if D["type"] == "cc":
        D = bin2lin(D)
    D = dict(D)
    beta = np.asarray(D["beta"], dtype=float)
    varbeta = np.asarray(D["varbeta"], dtype=float)
    D["z"] = beta / np.sqrt(varbeta)
    names = list(FM.keys())
    base_keys = ["beta", "varbeta", "snp", "MAF", "position", "z"]

    def _base():
        cols = {k: np.asarray(D[k]) for k in base_keys if k in D}
        return pd.DataFrame(cols)

    if len(FM) == 1:
        return {names[0]: _base()}
    YY = (np.sum(np.asarray(D["N"], float) * D["sdY"] ** 2)
          if D["type"] == "quant"
          else np.sum(np.asarray(D["N"], float) * D["s"] * (1 - D["s"])))
    if mode == "allbutone":
        sigs = [[names[k] for k in range(len(names)) if k != i]
                for i in range(len(names))]
        labels = names
    else:
        sigs = [None] + [names[:i] for i in range(1, len(names))]
        labels = ["+".join(s) if s else "" for s in sigs]
    cond = {}
    ld = _ld_matrix(D)
    for lab, sig in zip(labels, sigs):
        if not sig:
            cond[lab] = _base()
        else:
            cond[lab] = est_cond(D, ld, YY, sig)
    return cond


def map_cond(D, LD, YY, sigsnps=None):
    """Find the next lead SNP after conditioning. Port of ``map_cond``."""
    if not sigsnps:
        beta = np.asarray(D["beta"], dtype=float)
        varbeta = np.asarray(D["varbeta"], dtype=float)
        Z = beta / np.sqrt(varbeta)
        wh = int(np.argmax(np.abs(Z)))
        return {D["snp"][wh]: Z[wh]}
    est = est_cond(D, LD, YY, sigsnps)
    Z = est["beta"].values / np.sqrt(est["varbeta"].values)
    wh = int(np.argmax(np.abs(Z)))
    return {est["snp"].iloc[wh]: Z[wh]}


def map_mask(D, LD, r2thr=0.01, sigsnps=None):
    """Find the next lead SNP by masking. Port of ``map_mask``."""
    snp = list(D["snp"])
    if "beta" in D and "varbeta" in D:
        beta = np.asarray(D["beta"], dtype=float)
        varbeta = np.asarray(D["varbeta"], dtype=float)
        z = beta / np.sqrt(varbeta)
    else:
        pv = np.asarray(D["pvalues"], dtype=float)
        z = norm.isf(pv / 2)
    ld = LD.values if isinstance(LD, pd.DataFrame) else np.asarray(LD, float)
    if isinstance(LD, pd.DataFrame):
        ld_idx = {s: i for i, s in enumerate(LD.columns)}
    else:
        ld_idx = {s: i for i, s in enumerate(snp)}
    use = np.ones(len(snp), dtype=bool)
    if sigsnps:
        sig_cols = [ld_idx[s] for s in sigsnps]
        friends = np.any(np.abs(ld[:, sig_cols]) > np.sqrt(r2thr), axis=1)
        use = use & ~friends
        if not use.any():
            return None
        imask = np.array([snp.index(s) for s in sigsnps])
        expectedz = ld[:, sig_cols] @ np.abs(z[imask])
        zdiff = np.abs(z[use]) - np.abs(expectedz[use])
    else:
        zdiff = np.abs(z)[use]
    wh = int(np.argmax(zdiff))
    usable_snp = list(np.array(snp)[use])
    usable_z = z[use]
    return {usable_snp[wh]: usable_z[wh]}


def find_best_signal(D):
    """Return the single most-significant SNP. Port of ``find.best.signal``."""
    if D["type"] == "cc" and D.get("method") == "cond":
        D = bin2lin(D)
    beta = np.asarray(D["beta"], dtype=float)
    varbeta = np.asarray(D["varbeta"], dtype=float)
    z = beta / np.sqrt(varbeta)
    wh = int(np.argmax(np.abs(z)))
    return {D["snp"][wh]: z[wh]}


def finemap_signals(D, LD=None, method="single", r2thr=0.01,
                    pthr=1e-6, maxhits=3):
    """Iteratively identify independent signals in one trait.

    Faithful port of ``coloc:::finemap.signals`` (the hit-finding part;
    ``return.pp`` is not implemented as it is unused by
    :func:`coloc_signals`).

    Parameters
    ----------
    D : dict
        The dataset.
    LD : pandas.DataFrame or ndarray, optional
        SNP correlation matrix (defaults to ``D["LD"]``).
    method : {"single", "mask", "cond"}
        Signal-finding strategy.
    r2thr : float
        LD r-squared threshold for masking.
    pthr : float
        P-value threshold below which a signal is accepted.
    maxhits : int
        Maximum number of signals to report.

    Returns
    -------
    dict
        Mapping lead-SNP id -> marginal Z.
    """
    if LD is None:
        LD = D.get("LD")
    if method == "cond":
        check_dataset(D, req=("N", "MAF", "beta", "varbeta"))
        check_ld(D, LD)
    else:
        check_dataset(D)
    D = dict(D)
    if D["type"] == "quant" and "sdY" not in D:
        D["sdY"] = sdY_est(D["varbeta"], D["MAF"], D["N"])
    zthr = norm.isf(pthr / 2)
    ld = _ld_matrix(D, LD)
    D["LD"] = ld
    Dwork = D
    if D["type"] == "cc" and method == "cond":
        Dwork = bin2lin(D)
        Dwork["LD"] = ld
    if Dwork["type"] == "quant":
        YY = np.sum(np.asarray(Dwork["N"], float) * Dwork["sdY"] ** 2)
    else:
        YY = np.sum(np.asarray(Dwork["N"], float)
                    * Dwork["s"] * (1 - Dwork["s"]))
    hits = {}
    while len(hits) < maxhits:
        if method == "mask":
            newhit = map_mask(Dwork, ld, r2thr,
                              sigsnps=list(hits.keys()))
        else:
            newhit = map_cond(Dwork, ld, YY, sigsnps=list(hits.keys()))
        if newhit is None or len(newhit) == 0:
            break
        (snp_id, zval), = newhit.items()
        if abs(zval) < zthr:
            break
        hits[snp_id] = zval
        if method == "single":
            break
    return hits


def _gethits(hits, i, mode):
    """Port of ``coloc:::gethits`` (0-based ``i``)."""
    hits = list(hits)
    if mode == "allbutone":
        ret = [h for k, h in enumerate(hits) if k != i]
    else:
        if i == 0:
            return []
        ret = hits[:i]
    return [h for h in ret if h != ""]


def coloc_process(obj, hits1=None, hits2=None, LD1=None, LD2=None,
                  r2thr=0.01, p1=1e-4, p2=1e-4, p12=1e-6,
                  mode="iterative"):
    """Combine a ``coloc.detail`` result over multiple signals.

    Faithful port of ``coloc:::coloc.process``.  When more than one hit
    is supplied for a trait, SNPs in LD with the *other* signals are
    masked out (their ``lABF`` set to ``-1.1``) before recomputing the
    posterior probabilities.

    Parameters
    ----------
    obj : dict
        Output of :func:`pycoloc.coloc_detail`.
    hits1, hits2 : sequence of str, optional
        Lead SNPs per trait.
    LD1, LD2 : pandas.DataFrame, optional
        LD matrices for masking.
    r2thr : float
        LD r-squared threshold for masking.
    p1, p2, p12 : float
        Priors.
    mode : {"iterative", "allbutone"}
        Conditioning scheme for choosing which hits to mask.

    Returns
    -------
    dict
        Keys ``summary`` (per-signal-pair PPs), ``results`` and ``priors``.
    """
    res = dict(obj)
    df = res["results"].copy()
    df3 = res["results.H3"].copy()
    df = df.rename(columns={
        "lABF.h1": "lbf1", "lABF.h2": "lbf2", "lABF.h4": "lbf4"})
    df3 = df3.rename(columns={"lABF.h3": "lbf3"})

    def _f(d, d3):
        lH0 = 0.0
        lH1 = np.log(p1) + logsum(d["lbf1"].values)
        lH2 = np.log(p2) + logsum(d["lbf2"].values)
        lH3 = np.log(p1) + np.log(p2) + logsum(d3["lbf3"].values)
        lH4 = np.log(p12) + logsum(d["lbf4"].values)
        allabf = np.array([lH0, lH1, lH2, lH3, lH4])
        best1 = d["snp"].iloc[int(np.argmax(d["lbf1"].values))]
        best2 = d["snp"].iloc[int(np.argmax(d["lbf2"].values))]
        best4 = d["snp"].iloc[int(np.argmax(d["lbf4"].values))]
        denom = logsum(allabf)
        pp = np.exp(allabf - denom)
        row = {"nsnps": len(d)}
        for i in range(5):
            row[f"PP.H{i}.abf"] = pp[i]
        row.update({"best1": best1, "best2": best2, "best4": best4})
        return pd.DataFrame([row])

    if hits1 is None:
        hits1 = [""]
    if hits2 is None:
        hits2 = [""]
    hits1 = list(hits1) if hits1 else [""]
    hits2 = list(hits2) if hits2 else [""]
    if len(hits1) <= 1 and len(hits2) <= 1:
        tmp = _f(df, df3)
        tmp["hit1"] = hits1[0]
        tmp["hit2"] = hits2[0]
        return {"summary": tmp, "results": res["results"],
                "priors": pd.Series({"p1": p1, "p2": p2, "p12": p12})}
    # multi-signal: masking via LD
    def _ld_r2(LD, hits):
        hh = [h for h in set(hits) if h != ""]
        sub = LD.loc[hh] ** 2
        return sub

    ldf1 = _ld_r2(LD1, hits1) if len(hits1) > 1 else None
    ldf2 = _ld_r2(LD2, hits2) if len(hits2) > 1 else None
    todo = [(i, j) for j in range(len(hits2)) for i in range(len(hits1))]
    cols0 = [c for c in ["snp", "position"] if c in res["results"].columns]
    newresult = res["results"][cols0].copy()
    summaries = []
    for r, (i, j) in enumerate(todo, start=1):
        drop1, drop2 = None, None
        if len(hits1) > 1:
            ih1 = _gethits(hits1, i, mode)
            if not ih1:
                drop1 = []
            else:
                ldout1 = ldf1.loc[ih1].max(axis=0)
                drop1 = list(ldout1.index[ldout1 > r2thr])
        if len(hits2) > 1:
            ih2 = _gethits(hits2, j, mode)
            if not ih2:
                drop2 = []
            else:
                ldout2 = ldf2.loc[ih2].max(axis=0)
                drop2 = list(ldout2.index[ldout2 > r2thr])
        dropsnps = set((drop1 or []) + (drop2 or []))
        d = df.copy()
        d3 = df3.copy()
        if drop1:
            m = d["snp"].isin(drop1)
            d.loc[m, ["lbf1", "lbf4"]] = -1.1
        if drop2:
            m = d["snp"].isin(drop2)
            d.loc[m, ["lbf2", "lbf4"]] = -1.1
        denom = logsum(d["lbf4"].values)
        newresult[f"SNP.PP.H4.row{r}"] = np.exp(d["lbf4"].values - denom)
        z1 = res["results"].get("z.df1")
        z2 = res["results"].get("z.df2")
        if z1 is not None:
            newresult[f"z.df1.row{r}"] = np.where(
                d["snp"].isin(drop1 or []), 0.0, z1.values)
        if z2 is not None:
            newresult[f"z.df2.row{r}"] = np.where(
                d["snp"].isin(drop2 or []), 0.0, z2.values)
        if drop1 or drop2:
            mm = (d3["snp1"].isin(drop1 or [])
                  | d3["snp2"].isin(drop2 or []))
            d3.loc[mm, "lbf3"] = 0.0
        tmp = _f(d, d3)
        tmp["hit1"] = hits1[i]
        tmp["hit2"] = hits2[j]
        summaries.append(tmp)
    summary = pd.concat(summaries, ignore_index=True)
    return {"summary": summary, "results": newresult,
            "priors": pd.Series({"p1": p1, "p2": p2, "p12": p12})}


def coloc_signals(dataset1, dataset2, MAF=None, LD=None, method="single",
                  mode="iterative", p1=1e-4, p2=1e-4, p12=None,
                  maxhits=3, r2thr=0.01, pthr=1e-6):
    """Multiple-signal colocalisation of two traits.

    Faithful port of ``coloc::coloc.signals``.  Supports the ``single``,
    ``cond`` (conditioning) and ``mask`` (masking) strategies.

    Parameters
    ----------
    dataset1, dataset2 : dict
        The two datasets.  A per-dataset ``method`` / ``LD`` may be set;
        otherwise the function arguments apply to both.
    MAF, LD : optional
        Attached to datasets lacking them.
    method : {"single", "cond", "mask"}
        Signal-finding strategy.
    mode : {"iterative", "allbutone"}
        Conditioning scheme.
    p1, p2, p12 : float
        Priors.  ``p12`` is mandatory (no default, as in coloc 5).
    maxhits : int
        Maximum signals per trait.
    r2thr : float
        LD r-squared threshold.
    pthr : float
        P-value threshold for accepting signals.

    Returns
    -------
    ColocABF
        ``summary`` (one row per signal pair, with ``hit1``/``hit2`` and
        ``PP.H0..H4.abf``), ``results`` and ``priors``.
    """
    if p12 is None:
        raise ValueError(
            "default value for p12 has been removed. please choose a "
            "value appropriate for your study"
        )
    dataset1 = dict(dataset1)
    dataset2 = dict(dataset2)
    if "MAF" not in dataset1 and MAF is not None:
        dataset1["MAF"] = MAF
    if "MAF" not in dataset2 and MAF is not None:
        dataset2["MAF"] = MAF
    if "LD" not in dataset1 and LD is not None:
        dataset1["LD"] = LD
    if "LD" not in dataset2 and LD is not None:
        dataset2["LD"] = LD
    if "method" not in dataset1:
        dataset1["method"] = method
    if "method" not in dataset2:
        dataset2["method"] = method

    for D, idx in ((dataset1, 1), (dataset2, 2)):
        if D["method"] == "cond":
            check_dataset(D, idx, req=("beta", "varbeta", "MAF"))
            check_ld(D, D["LD"])
            if D["type"] == "quant" and "sdY" not in D:
                D["sdY"] = sdY_est(D["varbeta"], D["MAF"], D["N"])
        else:
            check_dataset(D, idx)

    fm1 = finemap_signals(dataset1, method=dataset1["method"],
                          maxhits=maxhits, r2thr=r2thr, pthr=pthr)
    fm2 = finemap_signals(dataset2, method=dataset2["method"],
                          maxhits=maxhits, r2thr=r2thr, pthr=pthr)
    if not fm1:
        fm1 = find_best_signal(dataset1)
    if not fm2:
        fm2 = find_best_signal(dataset2)

    cond1 = cond2 = None
    X1 = X2 = {}
    if fm1 and dataset1["method"] == "cond":
        cond1 = est_all_cond(dataset1, fm1, mode=mode)
        X1 = {k: dataset1[k] for k in ("N", "sdY", "type", "s")
              if k in dataset1}
    if fm2 and dataset2["method"] == "cond":
        cond2 = est_all_cond(dataset2, fm2, mode=mode)
        X2 = {k: dataset2[k] for k in ("N", "sdY", "type", "s")
              if k in dataset2}

    m1, m2 = dataset1["method"], dataset2["method"]
    res = None
    if m1 in ("single", "mask") and m2 in ("single", "mask"):
        col = coloc_detail(dataset1, dataset2, p1=p1, p2=p2, p12=p12)
        res = coloc_process(col, hits1=list(fm1), hits2=list(fm2),
                            LD1=_ld_matrix(dataset1),
                            LD2=_ld_matrix(dataset2),
                            r2thr=r2thr, mode=mode, p12=p12,
                            p1=p1, p2=p2)
    elif m1 == "cond" and m2 == "cond":
        res = []
        for i, c1 in enumerate(cond1.values()):
            for j, c2 in enumerate(cond2.values()):
                col = coloc_detail(_cond_to_dataset(c1, X1),
                                   _cond_to_dataset(c2, X2),
                                   p1=p1, p2=p2, p12=p12)
                res.append(coloc_process(
                    col, hits1=[list(fm1)[i]], hits2=[list(fm2)[j]],
                    LD1=_ld_matrix(dataset1), LD2=_ld_matrix(dataset2),
                    p12=p12, p1=p1, p2=p2))
    elif m1 == "cond" and m2 in ("single", "mask"):
        res = []
        for i, c1 in enumerate(cond1.values()):
            col = coloc_detail(_cond_to_dataset(c1, X1), dataset2,
                               p1=p1, p2=p2, p12=p12)
            res.append(coloc_process(
                col, hits1=[list(fm1)[i]], hits2=list(fm2),
                LD1=_ld_matrix(dataset1), LD2=_ld_matrix(dataset2),
                r2thr=r2thr, p12=p12, p1=p1, p2=p2))
    elif m1 in ("single", "mask") and m2 == "cond":
        res = []
        for j, c2 in enumerate(cond2.values()):
            col = coloc_detail(dataset1, _cond_to_dataset(c2, X2),
                               p1=p1, p2=p2, p12=p12)
            res.append(coloc_process(
                col, hits1=list(fm1), hits2=[list(fm2)[j]],
                LD1=_ld_matrix(dataset1), LD2=_ld_matrix(dataset2),
                r2thr=r2thr, p12=p12, p1=p1, p2=p2))

    if m1 == "cond" or m2 == "cond":
        summary = pd.concat([r["summary"] for r in res], ignore_index=True)
        rowvars = ["SNP.PP.H4", "z.df1", "z.df2"]
        results = res[0]["results"].copy()
        results = results.rename(columns={
            v: f"{v}.row1" for v in rowvars if v in results.columns})
        for i in range(1, len(res)):
            cols = [c for c in res[i]["results"].columns
                    if "snp" in c or "SNP.PP.H4" in c
                    or "z.df1" in c or "z.df2" in c]
            thisone = res[i]["results"][cols].copy()
            thisone = thisone.rename(columns={
                v: f"{v}.row{i + 1}" for v in rowvars
                if v in thisone.columns})
            results = results.merge(thisone, on="snp", how="outer")
    else:
        summary = res["summary"]
        results = res["results"]

    d1 = pd.DataFrame({"hit1": list(fm1),
                       "hit1.margz": list(fm1.values())})
    res_df = summary.merge(d1, on="hit1")
    d2 = pd.DataFrame({"hit2": list(fm2),
                       "hit2.margz": list(fm2.values())})
    res_df = res_df.merge(d2, on="hit2")
    out = ColocABF(
        summary=res_df, results=results,
        priors=pd.Series({"p1": p2, "p2": p2, "p12": p12}),
    )
    return out


def _cond_to_dataset(cond_df, extras):
    """Turn a conditioned-estimate DataFrame back into a dataset dict."""
    d = {c: cond_df[c].values for c in cond_df.columns}
    d.update(extras)
    return d
