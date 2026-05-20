"""pycoloc: Pure-Python port of the R/CRAN package *coloc*.

A faithful, dependency-light Python implementation of the Bayesian
colocalisation toolkit ``coloc`` (Giambartolomei *et al.*, *PLoS Genet.*
2014; Wallace, *PLoS Genet.* 2020 / 2021).  It tests whether two genetic
traits (e.g. a GWAS trait and a molecular QTL) share one or more causal
variants in a genomic region, using Approximate Bayes Factors (Wakefield
2009).

Only ``numpy`` / ``scipy`` / ``pandas`` are required; ``matplotlib`` is
needed for the plots.  There is **no** dependency on R or rpy2, and the
SuSiE-based functions consume pre-computed SuSiE log-Bayes-factor
matrices rather than running SuSiE themselves.

Datasets
--------
A *dataset* is a plain :class:`dict` of GWAS summary statistics for one
trait.  Recognised keys (mirroring coloc): ``beta``, ``varbeta``,
``pvalues``, ``MAF``, ``snp``, ``position``, ``N``, ``type``
(``"quant"`` / ``"cc"``), ``s`` (case proportion), ``sdY`` and ``LD``.

Single-causal-variant colocalisation
------------------------------------
* :func:`coloc_abf`     — the workhorse colocalisation function.
* :func:`coloc_detail`  — ``coloc.abf`` plus the all-pairs H3 grid.
* :func:`finemap_abf`   — single-trait ABF fine-mapping.
* :func:`combine_abf`   — combine per-SNP log-ABFs into PP.H0..H4.

Multiple causal variants
------------------------
* :func:`coloc_susie`   — colocalisation from SuSiE ``lbf_variable``.
* :func:`coloc_bf_bf`   — colocalise two log-Bayes-factor matrices.
* :func:`finemap_susie`, :func:`finemap_bf` — single-trait analogues.
* :func:`coloc_signals` — conditioning / masking multi-signal coloc.
* :func:`finemap_signals` — iterative independent-signal detection.

Sensitivity & plots
-------------------
* :func:`sensitivity`   — prior-sensitivity analysis + plot.
* :func:`plot_coloc_abf`, :func:`plot_dataset`, :func:`manhattan`.

Helpers
-------
* :func:`process_dataset`, :func:`check_dataset`, :func:`check_alignment`,
  :func:`check_ld`, :func:`subset_dataset`.
* :func:`approx_bf_estimates`, :func:`approx_bf_p`.
* :func:`sdY_est`, :func:`logsum`, :func:`logdiff`, :func:`Var_data`,
  :func:`Var_data_cc`, :func:`prior_snp2hyp`, :func:`prior_adjust`,
  :func:`logbf_to_pp`, :func:`est_cond`, :func:`est_all_cond`,
  :func:`coloc_process`.
"""
from __future__ import annotations

from .abf import (
    ColocABF,
    coloc_abf,
    coloc_detail,
    combine_abf,
    finemap_abf,
    prior_adjust,
    prior_snp2hyp,
)
from .dataset import (
    approx_bf_estimates,
    approx_bf_p,
    check_alignment,
    check_dataset,
    check_ld,
    process_dataset,
    subset_dataset,
)
from .plot import manhattan, plot_coloc_abf, plot_dataset
from .sensitivity import sensitivity
from .signals import (
    bin2lin,
    coloc_process,
    coloc_signals,
    est_all_cond,
    est_cond,
    find_best_signal,
    finemap_signals,
    map_cond,
    map_mask,
)
from .susie import (
    coloc_bf_bf,
    coloc_susie,
    finemap_bf,
    finemap_susie,
    logbf_to_pp,
)
from .utils import (
    Var_data,
    Var_data_cc,
    adjust_prior,
    logdiff,
    logsum,
    sdY_est,
)

__version__ = "0.1.0"

__all__ = [
    # single-variant colocalisation
    "coloc_abf",
    "coloc_detail",
    "finemap_abf",
    "combine_abf",
    "ColocABF",
    # multiple causal variants
    "coloc_susie",
    "coloc_bf_bf",
    "finemap_susie",
    "finemap_bf",
    "logbf_to_pp",
    "coloc_signals",
    "finemap_signals",
    "coloc_process",
    "est_cond",
    "est_all_cond",
    "find_best_signal",
    "map_cond",
    "map_mask",
    "bin2lin",
    # sensitivity & plots
    "sensitivity",
    "plot_coloc_abf",
    "plot_dataset",
    "manhattan",
    # dataset helpers
    "process_dataset",
    "check_dataset",
    "check_alignment",
    "check_ld",
    "subset_dataset",
    "approx_bf_estimates",
    "approx_bf_p",
    # numerical helpers
    "sdY_est",
    "logsum",
    "logdiff",
    "Var_data",
    "Var_data_cc",
    "adjust_prior",
    "prior_snp2hyp",
    "prior_adjust",
]
