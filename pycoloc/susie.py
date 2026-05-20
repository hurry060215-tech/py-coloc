"""Colocalisation allowing multiple causal variants (SuSiE-based).

Ports coloc's SuSiE-aware colocalisation:

* :func:`logbf_to_pp`   — ``coloc:::logbf_to_pp``
* :func:`coloc_bf_bf`   — ``coloc:::coloc.bf_bf``
* :func:`finemap_bf`    — ``coloc:::finemap.bf``
* :func:`coloc_susie`   — ``coloc::coloc.susie``
* :func:`finemap_susie` — ``coloc:::finemap.susie``

R's ``coloc.susie`` consumes the output of ``susieR::runsusie`` — in
particular the per-single-effect log-Bayes-factor matrix
``lbf_variable`` (shape ``L x p``) and the credible-set indices
``sets$cs_index``.  This port does **not** run SuSiE: instead
:func:`coloc_susie` accepts either

* the ``lbf_variable`` arrays directly (with optional ``cs_index``), or
* susie-like :class:`dict` objects carrying ``lbf_variable`` / ``sets``.

The per-credible-set ABF colocalisation maths are identical to
``coloc.bf_bf``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .abf import combine_abf
from .utils import logsum

__all__ = [
    "logbf_to_pp",
    "coloc_bf_bf",
    "finemap_bf",
    "coloc_susie",
    "finemap_susie",
]


def logbf_to_pp(bf, pi, last_is_null):
    """Convert a log-Bayes-factor matrix into posterior probabilities.

    Faithful port of ``coloc:::logbf_to_pp``.

    Parameters
    ----------
    bf : ndarray, shape (L, p)
        Log Bayes factors (rows = single effects / credible sets).
    pi : float or array_like
        Per-variant prior(s).  A scalar is capped at ``1/n`` and expanded.
    last_is_null : bool
        Whether the final column is a null model.

    Returns
    -------
    ndarray, shape (L, p)
        Posterior probabilities per row.
    """
    bf = np.asarray(bf, dtype=float)
    ncol = bf.shape[1]
    n = ncol - 1 if last_is_null else ncol
    pi = np.atleast_1d(np.asarray(pi, dtype=float))
    if pi.size == 1:
        p = float(pi[0])
        if p > 1.0 / n:
            p = 1.0 / n
        if last_is_null:
            pi = np.array(list(np.repeat(p, n)) + [1 - n * p])
        else:
            pi = np.repeat(p, n)
    if np.any(pi == 0):
        pi = pi.copy()
        pi[pi == 0] = 1e-16
        pi = pi / pi.sum()
    if last_is_null:
        bf = bf - bf[:, [-1]]
    priors = np.tile(np.log(pi), (bf.shape[0], 1))
    denom = np.array([logsum(row) for row in (bf + priors)])[:, None]
    return np.exp(bf + priors - denom)


def _as_bf_frame(bf, snps=None):
    """Coerce a 1-D/2-D lbf input into a (matrix, colnames) pair."""
    bf = np.asarray(bf, dtype=float)
    if bf.ndim == 1:
        bf = bf[None, :]
    p = bf.shape[1]
    if snps is None:
        snps = [f"snp{i + 1}" for i in range(p)]
    else:
        snps = list(snps)
    return bf, snps


def coloc_bf_bf(bf1, bf2, snps1=None, snps2=None,
                p1=1e-4, p2=1e-4, p12=5e-6,
                overlap_min=0.5, trim_by_posterior=True):
    """Colocalise two sets of log-Bayes-factor rows.

    Faithful port of ``coloc:::coloc.bf_bf``.  Each row of ``bf1`` /
    ``bf2`` is one credible set / single effect; every pair of rows is
    colocalised with the single-variant ABF maths.

    Parameters
    ----------
    bf1, bf2 : ndarray
        Log-Bayes-factor matrices (``L x p``) for the two traits.
    snps1, snps2 : sequence of str, optional
        SNP names labelling the columns.  Required for overlapping the
        two traits when the SNP sets differ.
    p1, p2, p12 : float
        Prior probabilities.
    overlap_min : float
        Minimum posterior mass overlap required to keep a signal pair.
    trim_by_posterior : bool
        Drop signal pairs with insufficient overlap.

    Returns
    -------
    dict
        Keys ``summary`` (per-pair PP.H0..H4 with ``idx1``/``idx2``),
        ``results`` (per-SNP ``SNP.PP.H4``) and ``priors``.
    """
    bf1, cn1 = _as_bf_frame(bf1, snps1)
    bf2, cn2 = _as_bf_frame(bf2, snps2)
    todo = [(i, j) for j in range(bf2.shape[0])
            for i in range(bf1.shape[0])]
    # expand.grid in R varies i fastest; replicate that ordering
    todo = sorted(todo, key=lambda t: (t[1], t[0]))
    isnps = [s for s in cn1 if s in set(cn2) and s != "null"]
    if not isnps:
        return {"summary": pd.DataFrame({"nsnps": [np.nan]})}
    if "null" in cn1:
        ni = cn1.index("null")
        bf1 = bf1 - bf1[:, [ni]]
    if "null" in cn2:
        ni = cn2.index("null")
        bf2 = bf2 - bf2[:, [ni]]
    pp1 = logbf_to_pp(bf1, p1, last_is_null="null" in cn1)
    pp2 = logbf_to_pp(bf2, p2, last_is_null="null" in cn2)
    pp1_df = pd.DataFrame(pp1, columns=cn1)
    pp2_df = pd.DataFrame(pp2, columns=cn2)
    ph0_1 = (pp1_df["null"].values if "null" in cn1
             else 1 - pp1_df.sum(axis=1).values)
    ph0_2 = (pp2_df["null"].values if "null" in cn2
             else 1 - pp2_df.sum(axis=1).values)
    nn1 = [c for c in cn1 if c != "null"]
    nn2 = [c for c in cn2 if c != "null"]
    prop1 = (pp1_df[isnps].sum(axis=1).values
             / pp1_df[nn1].sum(axis=1).values)
    prop2 = (pp2_df[isnps].sum(axis=1).values
             / pp2_df[nn2].sum(axis=1).values)
    if trim_by_posterior:
        drop = np.array([
            (prop1[i] < overlap_min) or (prop2[j] < overlap_min)
            for (i, j) in todo
        ])
        if drop.all():
            import warnings

            warnings.warn(
                "snp overlap too small between datasets: too few snps "
                "with high posterior in one trait represented in other"
            )
            hit1 = [cn1[np.argmax(pp1[i])] for (i, j) in todo]
            hit2 = [cn2[np.argmax(pp2[j])] for (i, j) in todo]
            summary = pd.DataFrame({
                "nsnps": len(isnps),
                "hit1": hit1, "hit2": hit2,
                "PP.H0.abf": [min(ph0_1[i], ph0_2[j])
                              for (i, j) in todo],
                "PP.H1.abf": np.nan, "PP.H2.abf": np.nan,
                "PP.H3.abf": np.nan, "PP.H4.abf": np.nan,
                "idx1": [i + 1 for (i, j) in todo],
                "idx2": [j + 1 for (i, j) in todo],
            })
            return {"summary": summary}
        todo = [t for t, dr in zip(todo, drop) if not dr]
    # restrict bf matrices to intersecting snps
    if len(isnps) != bf1.shape[1]:
        keep = [cn1.index(s) for s in isnps]
        bf1 = bf1[:, keep]
    if len(isnps) != bf2.shape[1]:
        keep = [cn2.index(s) for s in isnps]
        bf2 = bf2[:, keep]
    results, PP = [], []
    for (i, j) in todo:
        b1 = bf1[i, :]
        b2 = bf2[j, :]
        internal = b1 + b2
        denom = logsum(internal)
        snp_pp = np.exp(internal - denom)
        pp_abf = combine_abf(b1, b2, p1, p2, p12, quiet=True)
        PP.append(snp_pp)
        hit1 = isnps[int(np.argmax(b1))]
        hit2 = isnps[int(np.argmax(b2))]
        row = {"nsnps": len(isnps), "hit1": hit1, "hit2": hit2}
        for k, v in pp_abf.items():
            row[k] = v
        results.append(row)
    summary = pd.DataFrame(results)
    summary["idx1"] = [i + 1 for (i, j) in todo]
    summary["idx2"] = [j + 1 for (i, j) in todo]
    PP = np.array(PP).T  # snp x pair
    if len(todo) > 1:
        cols = [f"SNP.PP.H4.row{k + 1}" for k in range(len(todo))]
    else:
        cols = ["SNP.PP.H4.abf"]
    results_df = pd.DataFrame(PP, columns=cols)
    results_df.insert(0, "snp", isnps)
    return {
        "summary": summary,
        "results": results_df,
        "priors": pd.Series({"p1": p1, "p2": p2, "p12": p12}),
    }


def finemap_bf(bf1, snps=None, p1=1e-4):
    """Single-trait fine-mapping from a log-Bayes-factor matrix.

    Faithful port of ``coloc:::finemap.bf``.

    Parameters
    ----------
    bf1 : ndarray
        Log-Bayes-factor matrix (``L x p``).
    snps : sequence of str, optional
        Column SNP names.
    p1 : float
        Prior probability a SNP is causal.

    Returns
    -------
    dict
        Keys ``summary``, ``results`` (per-SNP posterior) and ``priors``.
    """
    bf1, cn1 = _as_bf_frame(bf1, snps)
    isnps = [s for s in cn1 if s != "null"]
    if not isnps:
        return {"summary": pd.DataFrame({"nsnps": [np.nan]})}
    if "null" in cn1:
        ni = cn1.index("null")
        bf1 = bf1 - bf1[:, [ni]]
    keep = [cn1.index(s) for s in isnps]
    bf1 = bf1[:, keep]
    results, PP = [], []
    for k in range(bf1.shape[0]):
        b1 = bf1[k, :]
        denom = logsum(b1)
        snp_pp = np.exp(b1 - denom)
        PP.append(snp_pp)
        hit1 = isnps[int(np.argmax(b1))]
        results.append({"nsnps": len(isnps), "hit": hit1})
    summary = pd.DataFrame(results)
    PP = np.array(PP).T
    if bf1.shape[0] > 1:
        cols = [f"SNP.PP.row{k + 1}" for k in range(bf1.shape[0])]
    else:
        cols = ["SNP.PP.abf"]
    results_df = pd.DataFrame(PP, columns=cols)
    results_df.insert(0, "snp", isnps)
    return {
        "summary": summary,
        "results": results_df,
        "priors": pd.Series({"p1": p1}),
    }


def _extract_susie(obj):
    """Pull ``(lbf_variable, snp_names, cs_index)`` out of a susie input.

    Accepts either an ``(L, p)`` array (optionally with separate snp names)
    or a susie-like dict with ``lbf_variable`` and ``sets``.
    """
    if isinstance(obj, dict) and "lbf_variable" in obj:
        lbf = np.asarray(obj["lbf_variable"], dtype=float)
        snps = obj.get("snp")
        if snps is None and "colnames" in obj:
            snps = obj["colnames"]
        sets = obj.get("sets", {})
        cs_index = sets.get("cs_index") if isinstance(sets, dict) else None
        return lbf, snps, cs_index
    lbf = np.asarray(obj, dtype=float)
    if lbf.ndim == 1:
        lbf = lbf[None, :]
    return lbf, None, None


def coloc_susie(dataset1, dataset2, snps1=None, snps2=None,
                cs_index1=None, cs_index2=None,
                p1=1e-4, p2=1e-4, p12=5e-6, **kwargs):
    """Colocalisation allowing multiple causal variants.

    Faithful port of ``coloc::coloc.susie``.  This port does **not** run
    SuSiE; it consumes the SuSiE log-Bayes-factor matrix ``lbf_variable``
    (shape ``L x p``) for each trait — exactly the object produced by
    ``susieR::runsusie``.

    Parameters
    ----------
    dataset1, dataset2 : ndarray or dict
        Either the ``lbf_variable`` matrix for each trait, or a
        susie-like dict carrying ``lbf_variable`` and (optionally)
        ``sets`` with ``cs_index``.
    snps1, snps2 : sequence of str, optional
        Column SNP names for each ``lbf_variable``.
    cs_index1, cs_index2 : sequence of int, optional
        1-based credible-set row indices to keep (overrides any
        ``cs_index`` carried by a dict input).
    p1, p2, p12 : float
        Prior probabilities.
    **kwargs
        Passed through to :func:`coloc_bf_bf`
        (``overlap_min``, ``trim_by_posterior``).

    Returns
    -------
    dict
        Per-credible-set-pair PP.H0..H4 (see :func:`coloc_bf_bf`).
    """
    lbf1, s1, ci1 = _extract_susie(dataset1)
    lbf2, s2, ci2 = _extract_susie(dataset2)
    if snps1 is not None:
        s1 = snps1
    if snps2 is not None:
        s2 = snps2
    if cs_index1 is not None:
        ci1 = cs_index1
    if cs_index2 is not None:
        ci2 = cs_index2
    # subset to credible-set rows (1-based indices, as in R)
    if ci1 is not None and len(ci1):
        bf1 = lbf1[np.asarray(ci1, dtype=int) - 1, :]
        cs1 = list(ci1)
    else:
        bf1 = lbf1
        cs1 = list(range(1, lbf1.shape[0] + 1))
    if ci2 is not None and len(ci2):
        bf2 = lbf2[np.asarray(ci2, dtype=int) - 1, :]
        cs2 = list(ci2)
    else:
        bf2 = lbf2
        cs2 = list(range(1, lbf2.shape[0] + 1))
    if bf1.shape[0] == 0 or bf2.shape[0] == 0:
        return {"summary": pd.DataFrame({"nsnps": [np.nan]})}
    ret = coloc_bf_bf(bf1, bf2, snps1=s1, snps2=s2,
                      p1=p1, p2=p2, p12=p12, **kwargs)
    if "idx1" in ret.get("summary", pd.DataFrame()).columns:
        # remap idx back to original credible-set indices, as R does
        ret["summary"]["idx1"] = [cs1[i - 1]
                                  for i in ret["summary"]["idx1"]]
        ret["summary"]["idx2"] = [cs2[j - 1]
                                  for j in ret["summary"]["idx2"]]
    return ret


def finemap_susie(dataset1, snps=None, cs_index=None, p1=1e-4):
    """Single-trait fine-mapping from a SuSiE ``lbf_variable`` matrix.

    Port of ``coloc:::finemap.susie``.

    Parameters
    ----------
    dataset1 : ndarray or dict
        The ``lbf_variable`` matrix or a susie-like dict.
    snps : sequence of str, optional
        Column SNP names.
    cs_index : sequence of int, optional
        1-based credible-set row indices to keep.
    p1 : float
        Prior probability a SNP is causal.

    Returns
    -------
    dict
        See :func:`finemap_bf`, with an extra ``idx`` summary column.
    """
    lbf, s, ci = _extract_susie(dataset1)
    if snps is not None:
        s = snps
    if cs_index is not None:
        ci = cs_index
    if ci is not None and len(ci):
        bf1 = lbf[np.asarray(ci, dtype=int) - 1, :]
        idx = list(ci)
    else:
        bf1 = lbf
        idx = list(range(1, lbf.shape[0] + 1))
    if bf1.shape[0] == 0:
        return {"summary": pd.DataFrame({"nsnps": [np.nan]})}
    ret = finemap_bf(bf1, snps=s, p1=p1)
    ret["summary"]["idx"] = idx
    return ret
