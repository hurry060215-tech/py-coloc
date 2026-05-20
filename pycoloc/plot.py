"""Plotting utilities for coloc results (matplotlib).

Ports the coloc plot functions:

* :func:`plot_coloc_abf` — ``coloc:::plot.coloc_abf`` (per-SNP |z| coloured
  by posterior probability, faceted by trait).
* :func:`plot_dataset`   — ``coloc:::plot_dataset`` (a Manhattan plot of a
  single dataset with optional highlighted credible sets).
* :func:`manhattan`      — a convenience per-SNP ``SNP.PP.H4`` Manhattan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = ["plot_coloc_abf", "plot_dataset", "manhattan"]


def plot_coloc_abf(obj, ax=None):
    """Plot a :func:`pycoloc.coloc_abf` result.

    Faithful port of ``coloc:::plot.coloc_abf``.  Draws per-SNP ``|z|``
    against position, one panel per trait, with point colour/size scaled
    by the per-SNP posterior probability ``SNP.PP.H4``.

    Parameters
    ----------
    obj : ColocABF or dict
        A coloc result carrying a ``results`` per-SNP table.
    ax : sequence of matplotlib axes, optional
        Two axes (one per trait) to draw into.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    res = obj["results"].copy()
    if "position" not in res.columns:
        res["position"] = np.arange(1, len(res) + 1)
    zcols = [c for c in res.columns if c.startswith("z.df")]
    ppcols = [c for c in res.columns if "PP" in c and "SNP" in c]
    if not zcols:
        raise ValueError("results has no z.df* columns to plot")
    pp = (res[ppcols[0]].values if ppcols
          else res.get("SNP.PP.H4", pd.Series(np.zeros(len(res)))).values)
    if ax is None:
        fig, ax = plt.subplots(len(zcols), 1, figsize=(7, 3 * len(zcols)),
                               squeeze=False)
        ax = ax.ravel()
    else:
        fig = ax[0].figure
    for k, zc in enumerate(zcols):
        sc = ax[k].scatter(res["position"], np.abs(res[zc].values),
                           c=pp, cmap="viridis", s=20 + 60 * pp,
                           vmin=0, vmax=1)
        ax[k].set_xlabel("position")
        ax[k].set_ylabel(f"|{zc}|")
        ax[k].set_title(zc)
        fig.colorbar(sc, ax=ax[k], label="SNP.PP.H4")
    fig.tight_layout()
    return fig


def plot_dataset(d, susie_obj=None, highlight_list=None, alty=None,
                 ylab="-log10(p)", show_legend=True, ax=None, **kwargs):
    """Manhattan plot of a single dataset.

    Faithful port of ``coloc:::plot_dataset``.

    Parameters
    ----------
    d : dict
        A coloc dataset with ``position`` and ``beta`` / ``varbeta``.
    susie_obj : dict, optional
        A susie-like object whose credible sets are highlighted.
    highlight_list : sequence of sequences, optional
        Explicit lists of SNP ids to highlight.
    alty : array_like, optional
        Use these y values instead of ``-log10(p)``.
    ylab : str
        Y-axis label.
    show_legend : bool
        Show a legend for the highlighted sets.
    ax : matplotlib axis, optional
        Axis to draw into.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if "position" not in d:
        raise ValueError("no position element given")
    if alty is None:
        beta = np.asarray(d["beta"], dtype=float)
        varbeta = np.asarray(d["varbeta"], dtype=float)
        z = beta / np.sqrt(varbeta)
        y = -(norm.logcdf(-np.abs(z)) + np.log(2)) / np.log(10)
    else:
        y = np.asarray(alty, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    pos = np.asarray(d["position"])
    ax.scatter(pos, y, color="grey", s=16, **kwargs)
    ax.set_xlabel("Position")
    ax.set_ylabel(ylab)
    palette = ["dodgerblue", "green", "#6A3D9A", "#FF7F00", "gold",
               "skyblue", "#FB9A99", "palegreen", "#CAB2D6"]
    if susie_obj is not None:
        sets = susie_obj.get("sets", {})
        cs = sets.get("cs", {})
        highlight_list = [list(v) for v in cs.values()]
    if highlight_list:
        snp = list(d["snp"])
        for i, hl in enumerate(highlight_list):
            w = [k for k, s in enumerate(snp) if s in set(hl)]
            ax.scatter(pos[w], y[w], facecolors="none",
                       edgecolors=palette[i % len(palette)], s=80)
        if show_legend:
            ax.legend([str(i + 1) for i in range(len(highlight_list))])
    return ax


def manhattan(obj, ax=None):
    """Per-SNP ``SNP.PP.H4`` Manhattan plot for a coloc result.

    Plots ``-log10(p)`` (derived from ``z.df1``) against position,
    coloured by the per-SNP posterior probability of H4.

    Parameters
    ----------
    obj : ColocABF or dict
        A coloc result.
    ax : matplotlib axis, optional
        Axis to draw into.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    res = obj["results"].copy()
    if "position" not in res.columns:
        res["position"] = np.arange(1, len(res) + 1)
    zc = next((c for c in res.columns if c.startswith("z.df")), None)
    if zc is None:
        raise ValueError("results has no z.df* column")
    z = res[zc].values
    logp = -(norm.logcdf(-np.abs(z)) + np.log(2)) / np.log(10)
    pp = res.get("SNP.PP.H4", pd.Series(np.zeros(len(res)))).values
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    sc = ax.scatter(res["position"], logp, c=pp, cmap="Blues",
                    edgecolors="gray", vmin=0, vmax=1, s=25)
    ax.figure.colorbar(sc, ax=ax, label="SNP.PP.H4")
    ax.set_xlabel("position")
    ax.set_ylabel("-log10(p)")
    ax.set_title("coloc per-SNP posterior")
    return ax
