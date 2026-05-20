"""Dataset checking and per-SNP Bayes-factor processing.

Ports ``coloc:::check_dataset``, ``coloc:::check_alignment``,
``coloc:::check_ld``, ``coloc:::process.dataset``,
``coloc:::approx.bf.estimates`` and ``coloc:::approx.bf.p`` from coloc.

A *dataset* is a plain :class:`dict` carrying GWAS summary statistics for a
single trait.  The recognised keys mirror coloc exactly:

``beta``, ``varbeta``, ``pvalues``, ``MAF``, ``snp``, ``position``, ``N``,
``type`` (``"quant"`` or ``"cc"``), ``s`` (case proportion), ``sdY``,
``LD`` (a square correlation matrix), and additionally ``method`` for
:func:`pycoloc.coloc_signals`.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from .utils import Var_data, Var_data_cc, sdY_est

__all__ = [
    "check_dataset",
    "check_alignment",
    "check_ld",
    "process_dataset",
    "approx_bf_estimates",
    "approx_bf_p",
    "subset_dataset",
]

_RECOGNISED = (
    "beta", "varbeta", "pvalues", "MAF", "snp", "position",
    "N", "type", "s", "sdY", "LD",
)


def _as_array(x):
    return np.asarray(x)


def check_dataset(d, suffix="", req=("type", "snp"), warn_minp=1e-6):
    """Validate a coloc dataset dictionary.

    Faithful port of ``coloc:::check_dataset``.  Raises :class:`ValueError`
    on a malformed dataset and returns ``None`` (as the R function does)
    when the dataset is valid.

    Parameters
    ----------
    d : dict
        The dataset to check.
    suffix : str
        A label used in error messages (``1``, ``2`` or ``""``).
    req : tuple of str
        Required keys.
    warn_minp : float
        Warn if the smallest p-value exceeds this threshold.
    """
    if not isinstance(d, dict):
        raise ValueError(f"dataset {suffix}: is not a list")
    nd = [k for k in d if k in _RECOGNISED]
    missing = set(req) - set(nd)
    if missing:
        raise ValueError(
            f"dataset {suffix}: missing required element(s) "
            + ", ".join(sorted(missing))
        )
    for v in nd:
        val = d[v]
        if v in ("type", "N", "s", "sdY"):
            arr = np.atleast_1d(np.asarray(val))
        else:
            arr = np.atleast_1d(np.asarray(val))
        if arr.dtype.kind in "fc" and np.any(np.isnan(arr)):
            raise ValueError(f"dataset {suffix}: {v} contains missing values")
    if d["type"] not in ("quant", "cc"):
        raise ValueError(f"dataset {suffix}: type must be quant or cc")
    if "snp" in nd:
        snp = list(d["snp"])
        if len(snp) != len(set(snp)):
            raise ValueError(f"dataset {suffix}: duplicated snps found")
    if "MAF" in nd:
        maf = np.asarray(d["MAF"], dtype=float)
        if np.any(np.isnan(maf)) or np.any(maf <= 0) or np.any(maf >= 1):
            raise ValueError(
                f"dataset {suffix}: MAF should be a numeric, strictly "
                ">0 & <1"
            )
    # length consistency
    l = -1
    for v in [k for k in nd if k in
              ("pvalues", "MAF", "beta", "varbeta", "snp", "position")]:
        ln = len(np.atleast_1d(np.asarray(d[v])))
        if l < 0:
            l = ln
        elif ln != l:
            raise ValueError(
                f"dataset {suffix}: lengths of inputs don't match: "
            )
    if "s" in nd:
        s = d["s"]
        if not np.isscalar(s) or s <= 0 or s >= 1:
            raise ValueError(f"dataset {suffix}: s must be between 0 and 1")
    if not ("beta" in nd and "varbeta" in nd):
        if not ("pvalues" in nd and "MAF" in nd):
            raise ValueError(
                f"dataset {suffix}: require p values and MAF if beta, "
                "varbeta are unavailable"
            )
        if np.any(np.asarray(d["pvalues"], dtype=float) <= 0):
            raise ValueError("pvalues should not be negative or exactly 0")
        if d["type"] == "cc" and "s" not in nd:
            raise ValueError(
                f"dataset {suffix}: require, s, proportion of samples who "
                "are cases, if beta, varbeta are unavailable"
            )
        if "N" not in nd or d.get("N") is None or np.any(
            np.asarray(d["N"], dtype=float) <= 0
        ):
            raise ValueError(f"dataset {suffix}: sample size N <=0 or not set")
        p = np.asarray(d["pvalues"], dtype=float)
    else:
        beta = np.asarray(d["beta"], dtype=float)
        varbeta = np.asarray(d["varbeta"], dtype=float)
        p = norm.cdf(-np.abs(beta / np.sqrt(varbeta))) * 2
    if np.min(p) > warn_minp:
        warnings.warn(
            f"minimum p value is: {np.min(p):.3g}\n"
            "If this is not as small as you expected, please check you "
            "supplied var(beta) and not sd(beta) for varbeta."
        )
    if d["type"] == "quant" and "sdY" not in nd:
        if not ("MAF" in nd and "N" in nd):
            raise ValueError(
                f"dataset {suffix}: must give sdY for type quant, or, if "
                "sdY unknown, MAF and N so it can be estimated"
            )
    if "LD" in nd:
        ld = d["LD"]
        ld_arr = np.asarray(ld)
        if ld_arr.shape[0] != ld_arr.shape[1]:
            raise ValueError("LD not square")
        if isinstance(ld, pd.DataFrame):
            if list(ld.columns) != list(ld.index):
                raise ValueError("LD rownames != colnames")
            if set(d.get("snp", [])) - set(ld.columns):
                raise ValueError("colnames in LD do not contain all SNPs")
    return None


def check_ld(D, LD):
    """Port of ``coloc:::check_ld`` — validate an LD matrix for a dataset."""
    if LD is None:
        raise ValueError("LD required")
    arr = np.asarray(LD) if not isinstance(LD, pd.DataFrame) else LD.values
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("LD not square")
    if isinstance(LD, pd.DataFrame):
        if list(LD.columns) != list(LD.index):
            raise ValueError("LD rownames != colnames")
        if set(D.get("snp", [])) - set(LD.columns):
            raise ValueError("colnames in LD do not contain all SNPs")
    return None


def check_alignment(D, thr=0.2, do_plot=False, ax=None):
    """Check sign alignment between Z scores and LD.

    Port of ``coloc:::check_alignment``.  Returns the proportion of
    ``z_i z_j / r_ij`` values that are positive for SNP pairs with
    ``|r| > thr``.  Most should be positive when alleles are aligned.

    Parameters
    ----------
    D : dict
        A dataset with ``beta``, ``varbeta`` and ``LD``.
    thr : float
        LD threshold for which pairs are considered.
    do_plot : bool
        Draw a histogram of the ratio statistic.
    ax : matplotlib axis, optional
        Axis to draw into.

    Returns
    -------
    float
        Fraction of positive ratios.
    """
    check_dataset(D)
    beta = np.asarray(D["beta"], dtype=float)
    varbeta = np.asarray(D["varbeta"], dtype=float)
    z = beta / np.sqrt(varbeta)
    ld = D["LD"]
    ld = ld.values if isinstance(ld, pd.DataFrame) else np.asarray(ld, float)
    bprod = np.outer(z, z)
    mask = np.abs(ld) > thr
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (bprod / ld)[mask]
    ratio = ratio[np.isfinite(ratio)]
    frac = float(np.mean(ratio > 0))
    if do_plot:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.hist(ratio, bins=40)
        ax.axvline(0, color="red")
        ax.set_xlabel("ratio of product of Z scores to LD")
        ax.set_title("alignment check plot")
        ax.legend([f"% positive = {100 * round(frac, 3)}"])
    return frac


def approx_bf_estimates(z, V, type, suffix=None, sdY=1.0):
    """Per-SNP log approximate Bayes factor from effect estimates.

    Port of ``coloc:::approx.bf.estimates``.

    Parameters
    ----------
    z : array_like
        Z statistics (``beta / sqrt(varbeta)``).
    V : array_like
        Variance of beta (``varbeta``).
    type : str
        ``"quant"`` or ``"cc"``.
    suffix : str, optional
        Suffix appended to the output column names.
    sdY : float
        Phenotype standard deviation (quant only).

    Returns
    -------
    pandas.DataFrame
        Columns ``V``, ``z``, ``r``, ``lABF`` (suffixed if requested).
    """
    z = np.asarray(z, dtype=float)
    V = np.asarray(V, dtype=float)
    sd_prior = 0.15 * sdY if type == "quant" else 0.2
    r = sd_prior ** 2 / (sd_prior ** 2 + V)
    lABF = 0.5 * (np.log(1 - r) + (r * z ** 2))
    ret = pd.DataFrame({"V": V, "z": z, "r": r, "lABF": lABF})
    if suffix is not None:
        ret.columns = [f"{c}.{suffix}" for c in ret.columns]
    return ret


def approx_bf_p(p, f, type, N, s=None, suffix=None):
    """Per-SNP log approximate Bayes factor from p-values and MAF.

    Port of ``coloc:::approx.bf.p``.

    Parameters
    ----------
    p : array_like
        P-values.
    f : array_like
        Minor allele frequencies.
    type : str
        ``"quant"`` or ``"cc"``.
    N : array_like
        Sample size.
    s : float, optional
        Case proportion (required for ``type="cc"``).
    suffix : str, optional
        Suffix appended to the output column names.

    Returns
    -------
    pandas.DataFrame
        Columns ``V``, ``z``, ``r``, ``lABF`` (suffixed if requested).
    """
    p = np.asarray(p, dtype=float)
    f = np.asarray(f, dtype=float)
    if type == "quant":
        sd_prior = 0.15
        V = Var_data(f, N)
    else:
        sd_prior = 0.2
        V = Var_data_cc(f, N, s)
    z = norm.isf(0.5 * p)  # qnorm(0.5*p, lower.tail=FALSE)
    r = sd_prior ** 2 / (sd_prior ** 2 + V)
    lABF = 0.5 * (np.log(1 - r) + (r * z ** 2))
    ret = pd.DataFrame({"V": V, "z": z, "r": r, "lABF": lABF})
    if suffix is not None:
        ret.columns = [f"{c}.{suffix}" for c in ret.columns]
    return ret


def process_dataset(d, suffix):
    """Turn a dataset dictionary into a per-SNP Bayes-factor table.

    Port of ``coloc:::process.dataset``.  Uses the effect-estimate path
    when ``beta`` and ``varbeta`` are present, otherwise the p-value path.

    Parameters
    ----------
    d : dict
        A coloc dataset.
    suffix : str
        Suffix for output columns (e.g. ``"df1"``).

    Returns
    -------
    pandas.DataFrame
        Per-SNP table with at least a ``snp`` column and a suffixed
        ``lABF.<suffix>`` column.
    """
    nd = set(d)
    if "beta" in nd and "varbeta" in nd:
        d = dict(d)
        if d["type"] == "quant" and "sdY" not in nd:
            d["sdY"] = sdY_est(d["varbeta"], d["MAF"], d["N"])
        beta = np.asarray(d["beta"], dtype=float)
        varbeta = np.asarray(d["varbeta"], dtype=float)
        df = approx_bf_estimates(
            z=beta / np.sqrt(varbeta), V=varbeta,
            type=d["type"], suffix=suffix, sdY=d.get("sdY", 1.0),
        )
        df["snp"] = [str(x) for x in d["snp"]]
        if "position" in nd:
            df["position"] = np.asarray(d["position"])
        return df
    if "pvalues" in nd and "MAF" in nd and "N" in nd:
        df = pd.DataFrame({
            "pvalues": np.asarray(d["pvalues"], dtype=float),
            "MAF": np.asarray(d["MAF"], dtype=float),
            "N": np.broadcast_to(np.asarray(d["N"], dtype=float),
                                 (len(d["snp"]),)).copy(),
            "snp": [str(x) for x in d["snp"]],
        })
        df.columns = [c if c == "snp" else f"{c}.{suffix}"
                      for c in df.columns]
        abf = approx_bf_p(
            p=df[f"pvalues.{suffix}"].values, f=df[f"MAF.{suffix}"].values,
            type=d["type"], N=df[f"N.{suffix}"].values,
            s=d.get("s"), suffix=suffix,
        )
        df = pd.concat([df.reset_index(drop=True),
                        abf.reset_index(drop=True)], axis=1)
        if "position" in nd:
            df["position"] = np.asarray(d["position"])
        return df
    raise ValueError(
        "Must give, as a minimum, one of:\n"
        "(beta, varbeta, type, sdY)\n(beta, varbeta, type, MAF)\n"
        "(pvalues, MAF, N, type)"
    )


def subset_dataset(dataset, index):
    """Subset a dataset to a chosen set of SNP indices.

    Port of ``coloc:::subset_dataset``.  Vectors of SNP length and the
    square LD matrix are subset; scalar entries are left untouched.

    Parameters
    ----------
    dataset : dict
        A coloc dataset.
    index : array_like of int
        0-based indices of SNPs to keep.

    Returns
    -------
    dict
        The subset dataset.
    """
    index = np.asarray(index, dtype=int)
    if len(index) == 0:
        return dataset
    n = len(dataset["snp"])
    if n <= 1:
        raise ValueError("not trimming length 1 dataset.")
    if index.max() >= n:
        raise ValueError(
            "cannot subset to more than the number of snps in the dataset."
        )
    d = dict(dataset)
    for v, val in dataset.items():
        if isinstance(val, pd.DataFrame):
            if val.shape == (n, n):
                d[v] = val.iloc[index, index]
            continue
        arr = np.asarray(val)
        if arr.ndim == 2 and arr.shape == (n, n):
            d[v] = arr[np.ix_(index, index)]
        elif arr.ndim == 1 and len(arr) == n:
            d[v] = arr[index]
    return d
