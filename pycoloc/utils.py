"""Numerical helpers ported faithfully from coloc (R/claudia.R).

These small log-space utilities and variance formulae underpin every coloc
calculation.  They mirror the corresponding R functions one-to-one so that
results are numerically identical to the upstream package.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "logsum",
    "logdiff",
    "Var_data",
    "Var_data_cc",
    "sdY_est",
    "adjust_prior",
]


def logsum(x):
    """Log of the sum of exponentials, numerically stable.

    Port of ``coloc:::logsum``::

        my.max <- max(x)
        my.res <- my.max + log(sum(exp(x - my.max)))

    Parameters
    ----------
    x : array_like
        A vector of log values.

    Returns
    -------
    float
        ``log(sum(exp(x)))``.
    """
    x = np.asarray(x, dtype=float)
    my_max = np.max(x)
    return my_max + np.log(np.sum(np.exp(x - my_max)))


def logdiff(x, y):
    """Log of ``exp(x) - exp(y)``, numerically stable.

    Port of ``coloc:::logdiff``.  ``x`` and ``y`` are scalars (or
    broadcastable arrays); the result is computed in log space.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    my_max = np.maximum(np.max(x), np.max(y))
    res = my_max + np.log(np.exp(x - my_max) - np.exp(y - my_max))
    return float(res) if np.ndim(res) == 0 else res


def Var_data(f, N):
    """Variance of the estimated effect size for a quantitative trait.

    Port of ``coloc:::Var.data``: ``1 / (2 * N * f * (1 - f))``.
    """
    f = np.asarray(f, dtype=float)
    return 1.0 / (2.0 * N * f * (1.0 - f))


def Var_data_cc(f, N, s):
    """Variance of the estimated effect size for a case/control trait.

    Port of ``coloc:::Var.data.cc``::

        1 / (2 * N * f * (1 - f) * s * (1 - s))
    """
    f = np.asarray(f, dtype=float)
    return 1.0 / (2.0 * N * f * (1.0 - f) * s * (1.0 - s))


def sdY_est(vbeta, maf, n):
    """Estimate the standard deviation of a quantitative phenotype.

    Port of ``coloc:::sdY.est``.  Fits the no-intercept regression
    ``2 * n * maf * (1 - maf) ~ 1 / vbeta`` and returns the square root
    of the slope.

    Parameters
    ----------
    vbeta : array_like
        Variance of the regression coefficients (``varbeta``).
    maf : array_like
        Minor allele frequencies.
    n : int or array_like
        Sample size.

    Returns
    -------
    float
        Estimated ``sdY``.

    Raises
    ------
    ValueError
        If the estimated coefficient is negative.
    """
    vbeta = np.asarray(vbeta, dtype=float)
    maf = np.asarray(maf, dtype=float)
    oneover = 1.0 / vbeta
    nvx = 2.0 * np.asarray(n, dtype=float) * maf * (1.0 - maf)
    # no-intercept least squares: cf = sum(x*y) / sum(x*x)
    cf = np.sum(oneover * nvx) / np.sum(oneover * oneover)
    if cf < 0:
        raise ValueError(
            "estimated sdY is negative - this can happen with small "
            "datasets, or those with errors. A reasonable estimate of "
            "sdY is required to continue."
        )
    return float(np.sqrt(cf))


def adjust_prior(p, nsnps, suffix=""):
    """Cap a prior so that ``p * nsnps < 1``.

    Port of ``coloc:::adjust_prior``.  If ``nsnps * p >= 1`` the prior is
    reset to ``1 / (nsnps + 1)`` and a warning is issued.
    """
    if nsnps * p >= 1:
        import warnings

        warnings.warn(
            f"p{suffix} * nsnps >= 1, setting p{suffix}=1/(nsnps + 1)"
        )
        return 1.0 / (nsnps + 1)
    return p
