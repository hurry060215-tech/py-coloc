# py-coloc

A pure-Python port of the R/CRAN package
[**coloc**](https://cran.r-project.org/package=coloc) — Bayesian
**coloc**alisation of two genetic traits.

`pycoloc` tests whether two genetic traits (for example a GWAS trait and
a molecular QTL) share one or more causal variants in a genomic region,
using Approximate Bayes Factors.  It is a faithful re-implementation of
coloc 5.2.3 — the posterior probabilities match the R package to machine
precision (`< 1e-4` absolute, typically `< 1e-15`).

* Methods: Giambartolomei *et al.*, *PLoS Genet.* 2014 (single causal
  variant); Wallace, *PLoS Genet.* 2020 / 2021 (SuSiE & conditioning);
  Wakefield, *Genet. Epidemiol.* 2009 (Approximate Bayes Factors).
* Dependencies: `numpy`, `scipy`, `pandas`, `matplotlib` only.
  **No R, no rpy2.**

## Installation

```bash
pip install pycoloc
```

## The five posterior hypotheses

`coloc_abf` returns five posterior probabilities for a region:

| Hypothesis | Meaning                                            |
|------------|----------------------------------------------------|
| **PP.H0**  | no causal variant for either trait                 |
| **PP.H1**  | causal variant for trait 1 only                    |
| **PP.H2**  | causal variant for trait 2 only                    |
| **PP.H3**  | distinct causal variants for the two traits        |
| **PP.H4**  | a single, *shared* causal variant (colocalisation) |

## Quick start

A *dataset* is a plain `dict` of GWAS summary statistics for one trait.

```python
import numpy as np
import pycoloc as cl

# two traits, each with one (shared) causal SNP
rng = np.random.default_rng(0)
n = 500
def make(causal):
    beta = rng.normal(0, 0.02, n)
    beta[causal] = 0.4
    return dict(
        snp=[f"s{i}" for i in range(n)],
        position=np.arange(n) * 1000,
        beta=beta, varbeta=np.full(n, 0.0025),
        MAF=rng.uniform(0.05, 0.5, n),
        type="quant", N=4000, sdY=1.0,
    )

D1, D2 = make(250), make(250)
res = cl.coloc_abf(D1, D2)
print(res["summary"])          # nsnps, PP.H0.abf ... PP.H4.abf
print(res["results"].head())   # per-SNP lABF and SNP.PP.H4
```

## Single-causal-variant colocalisation

```python
res  = cl.coloc_abf(D1, D2, p1=1e-4, p2=1e-4, p12=1e-5)
fm   = cl.finemap_abf(D1)          # single-trait ABF fine-mapping
det  = cl.coloc_detail(D1, D2)     # also returns the all-pairs H3 grid
```

## Multiple causal variants (SuSiE)

`coloc_susie` consumes the per-single-effect log-Bayes-factor matrix
`lbf_variable` (shape `L x p`) produced by `susieR::runsusie` — it does
**not** run SuSiE itself, so there is no SuSiE dependency.

```python
# lbf1, lbf2 : (L x p) arrays from runsusie(); cs_index : credible sets
res = cl.coloc_susie(lbf1, lbf2,
                     snps1=snp_names, snps2=snp_names,
                     cs_index1=[1], cs_index2=[1])
print(res["summary"])   # one row per credible-set pair, PP.H0..H4
```

You may also pass susie-like dicts carrying `lbf_variable` and `sets`.

## Multiple-signal colocalisation by conditioning / masking

```python
# requires beta, varbeta, MAF, N and an LD matrix in each dataset
res = cl.coloc_signals(D1, D2, method="cond", p12=1e-5)  # conditioning
res = cl.coloc_signals(D1, D2, method="mask", p12=1e-5)  # masking
```

## Prior-sensitivity analysis

```python
res  = cl.coloc_abf(D1, D2)
sens = cl.sensitivity(res, "H4 > 0.5", dataset1=D1, dataset2=D2)
```

`sensitivity` sweeps the `p12` prior, reports the range over which the
rule holds, and (with `doplot=True`) draws the four-panel coloc
sensitivity figure.

## Plots

```python
cl.plot_coloc_abf(res)   # per-SNP |z| coloured by SNP.PP.H4, per trait
cl.plot_dataset(D1)      # Manhattan plot of one dataset
cl.manhattan(res)        # per-SNP posterior Manhattan
```

## Public API

`coloc_abf`, `coloc_detail`, `finemap_abf`, `combine_abf`, `ColocABF`,
`coloc_susie`, `coloc_bf_bf`, `finemap_susie`, `finemap_bf`,
`logbf_to_pp`, `coloc_signals`, `finemap_signals`, `coloc_process`,
`est_cond`, `est_all_cond`, `find_best_signal`, `map_cond`, `map_mask`,
`bin2lin`, `sensitivity`, `plot_coloc_abf`, `plot_dataset`, `manhattan`,
`process_dataset`, `check_dataset`, `check_alignment`, `check_ld`,
`subset_dataset`, `approx_bf_estimates`, `approx_bf_p`, `sdY_est`,
`logsum`, `logdiff`, `Var_data`, `Var_data_cc`, `adjust_prior`,
`prior_snp2hyp`, `prior_adjust`.

## R parity

`tests/r_reference_driver.R` runs coloc 5.2.3 on the bundled
`coloc_test_data` and exports both the raw inputs and the reference
results; `tests/test_r_parity.py` then runs `pycoloc` on the identical
inputs.  On `coloc_test_data` (500 SNPs):

| Function        | Agreement vs coloc 5.2.3                          |
|-----------------|---------------------------------------------------|
| `coloc_abf`     | PP.H0..H4 max abs diff `< 3e-16`; SNP.PP.H4 r = 1 |
| `finemap_abf`   | SNP.PP Pearson r = 1 (quant and case/control)     |
| `coloc_susie`   | per-pair PP.H0..H4 max abs diff `< 4e-16`         |
| `coloc_signals` | PP.H0..H4 max abs diff `< 4e-16` (cond and mask)  |
| `sensitivity`   | swept posterior grid max abs diff `< 1e-6`        |

```bash
python -m pytest tests/ -q
```

## License

GPL-3, matching the upstream coloc package
(`License: GPL` in its DESCRIPTION).

## Citing

If you use `pycoloc`, please cite the original coloc methods papers:

* Giambartolomei C *et al.* (2014) Bayesian test for colocalisation
  between pairs of genetic association studies using summary statistics.
  *PLoS Genet.* 10(5):e1004383.
* Wallace C (2020) Eliciting priors and relaxing the single causal
  variant assumption in colocalisation analyses. *PLoS Genet.*
  16(4):e1008720.
* Wallace C (2021) A more accurate method for colocalisation analysis
  allowing for multiple causal variants. *PLoS Genet.* 17(9):e1009440.
