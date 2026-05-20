"""Prior-sensitivity analysis for coloc results.

Port of ``coloc::sensitivity`` and its plot.  Given a colocalisation
result and a decision rule (e.g. ``"H4 > 0.5"``), the function sweeps the
prior ``p12`` over a logarithmic grid, re-weights the posterior
probabilities with :func:`pycoloc.prior_adjust`, and reports / shades the
range of ``p12`` for which the rule holds.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .abf import ColocABF, prior_adjust, prior_snp2hyp

__all__ = ["sensitivity"]

_PP_NAMES = ["PP.H0.abf", "PP.H1.abf", "PP.H2.abf", "PP.H3.abf", "PP.H4.abf"]


def _rule_to_expr(rule):
    """Translate a user rule (``H4 > 0.5``) into a pandas-evaluable one."""
    return re.sub(r"(H[0-4])", r"PP.\1.abf", rule)


def _eval_rule(rule_expr, pp):
    """Evaluate the (possibly vectorised) decision rule against a row/frame.

    The dotted ``PP.H4.abf`` names are rewritten to underscore-safe
    identifiers; numeric literals (``0.5``) are left untouched.
    """
    env = {}
    keys = pp.columns if isinstance(pp, pd.DataFrame) else pp.index
    expr = rule_expr
    for c in keys:
        safe = c.replace(".", "_")
        expr = expr.replace(c, safe)
        env[safe] = pp[c].values if isinstance(pp, pd.DataFrame) else pp[c]
    return eval(expr, {"__builtins__": {}}, env)


def sensitivity(obj, rule="", dataset1=None, dataset2=None, npoints=100,
                doplot=True, plot_manhattans=True, row=1, axes=None):
    """Prior-sensitivity analysis over the ``p12`` prior.

    Faithful port of ``coloc::sensitivity``.

    Parameters
    ----------
    obj : ColocABF or dict
        A colocalisation result (must carry ``summary`` and ``priors``).
    rule : str
        Decision rule, e.g. ``"H4 > 0.5"`` or ``"H4 > 0.8 & H3 < 0.1"``.
    dataset1, dataset2 : dict, optional
        The original datasets — supplied to enable Manhattan panels.
    npoints : int
        Number of ``p12`` grid points.
    doplot : bool
        Draw the sensitivity figure.
    plot_manhattans : bool
        Include the per-trait Manhattan panels (needs ``dataset1/2``).
    row : int
        Which signal-pair row to analyse (1-based) for multi-signal
        results.
    axes : sequence of matplotlib axes, optional
        Pre-existing axes to draw into.

    Returns
    -------
    pandas.DataFrame
        Columns ``H0`` ... ``H4``, ``p12`` and a boolean ``pass`` for
        every grid point.
    """
    if not (isinstance(obj, dict) and "priors" in obj and "summary" in obj):
        raise ValueError("obj must carry 'priors' and 'summary'")
    if rule == "":
        raise ValueError(
            "please supply a rule to define colocalisation, "
            "eg 'H4 > thr'"
        )
    rule_init = rule
    rule_expr = _rule_to_expr(rule)

    summ = obj["summary"]
    results = obj.get("results")
    multiple = False
    if isinstance(summ, pd.DataFrame):
        if not (1 <= row <= len(summ)):
            raise ValueError(f"row must be between 1 and {len(summ)}")
        srow = summ.iloc[row - 1]
        pp = srow[[c for c in summ.columns
                   if "PP" in c or "nsnp" in c]]
        if results is not None and f"SNP.PP.H4.row{row}" in results.columns:
            multiple = True
            results = results.copy()
            results["SNP.PP.H4"] = results[f"SNP.PP.H4.row{row}"]
        if results is not None and f"z.df1.row{row}" in results.columns:
            results["z.df1"] = results[f"z.df1.row{row}"]
            results["z.df2"] = results[f"z.df2.row{row}"]
    else:
        pp = summ

    pr = obj["priors"]
    p12 = float(pr["p12"])
    p1 = float(pr["p1"])
    p2 = float(pr["p2"])

    pp_dict = {k: float(pp[k]) for k in _PP_NAMES}
    pp_dict["nsnps"] = float(pp["nsnps"])
    pass_init = bool(_eval_rule(rule_expr, pd.Series(pp_dict)))
    print(f"Results {'pass' if pass_init else 'fail'} "
          f"decision rule {rule_init}")

    testp12 = np.logspace(np.log10(p1 * p2),
                          np.log10(min(p1, p1)), npoints)
    testH = prior_snp2hyp(pp_dict["nsnps"], p12=testp12, p1=p1, p2=p2)
    testpp = prior_adjust(pd.Series(pp_dict), newp12=testp12,
                          p1=p1, p2=p2, p12=p12)
    testpp_named = testpp.copy()
    testpp_named.columns = [f"PP.{c}.abf" for c in testpp.columns]
    passv = np.atleast_1d(_eval_rule(rule_expr, testpp_named))
    passv = np.asarray(passv, dtype=bool)

    if doplot:
        _sensitivity_plot(testp12, testH, testpp, passv, rule_init,
                          p12, results, multiple, row,
                          dataset1, dataset2, plot_manhattans, axes)

    out = testpp.copy()
    out["p12"] = testp12
    out["pass"] = passv
    return out


def _sensitivity_plot(testp12, testH, testpp, passv, rule_init, p12,
                      results, multiple, row, dataset1, dataset2,
                      plot_manhattans, axes):
    """Draw the four-panel sensitivity figure (matplotlib)."""
    import matplotlib.pyplot as plt

    have_manh = plot_manhattans and dataset1 is not None \
        and dataset2 is not None
    colors = ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"]

    if axes is None:
        if have_manh:
            fig, axarr = plt.subplots(2, 2, figsize=(11, 8))
            axes = axarr.ravel(order="F")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig = axes[0].figure

    panel = 0
    if have_manh:
        _manh_plot(axes[0], dataset1, results, 1, multiple, row)
        _manh_plot(axes[1], dataset2, results, 2, multiple, row)
        panel = 2

    ms = [(testH, "Prior probabilities"),
          (testpp, "Posterior probabilities")]
    for k, (m, title) in enumerate(ms):
        ax = axes[panel + k]
        marr = m.values
        for h in range(5):
            ax.plot(testp12, marr[:, h], "-o", color=colors[h],
                    markersize=4, label=f"H{h}")
        ax.set_xscale("log")
        ax.set_xlabel("p12")
        ax.set_ylabel("Prob")
        ax.set_title(title, loc="left")
        ax.axvline(p12, ls="--", color="gray")
        if passv.any():
            w = np.where(passv)[0]
            ax.axvspan(testp12[w.min()], testp12[w.max()],
                       color="green", alpha=0.1)
        ax.text(0.0, -0.18, f"shaded region: {rule_init}",
                transform=ax.transAxes, fontsize=8)
        if k == 0:
            ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    return fig


def _manh_plot(ax, dataset, results, wh, multiple, row):
    """Per-trait Manhattan panel coloured by SNP.PP.H4 (port of manh.plot)."""
    from scipy.stats import norm

    znm = f"z.df{wh}"
    if results is not None and znm in results.columns:
        z = results[znm].values
        snp_pp = results.get("SNP.PP.H4")
        snp_pp = (snp_pp.values if snp_pp is not None
                  else np.zeros(len(z)))
        pos = (results["position"].values
               if "position" in results.columns
               else np.arange(len(z)))
    else:
        beta = np.asarray(dataset["beta"], float)
        varbeta = np.asarray(dataset["varbeta"], float)
        z = beta / np.sqrt(varbeta)
        snp_pp = np.zeros(len(z))
        pos = np.asarray(dataset.get("position", np.arange(len(z))))
    logp = -(norm.logcdf(-np.abs(z)) + np.log(2)) / np.log(10)
    sc = ax.scatter(pos, logp, c=snp_pp, cmap="Blues",
                    edgecolors="gray", vmin=0, vmax=1, s=25)
    ax.set_xlabel("Chromosome position")
    ax.set_ylabel("-log10(p)")
    suffix = f" row {row}" if multiple else ""
    ax.set_title(f"trait {wh}{suffix}")
    return sc
