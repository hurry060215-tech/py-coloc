#!/usr/bin/env Rscript
# Drive the R coloc package on its bundled coloc_test_data.
#
# Usage:
#   Rscript r_reference_driver.R <out_dir>
#
# Outputs (in out_dir):
#   D1.tsv, D2.tsv      the two datasets (beta/varbeta/snp/position/MAF)
#   D1_LD.tsv, D2_LD.tsv  the LD matrices
#   D_meta.tsv          dataset scalars (type, N, sdY)
#   coloc_abf_summary.tsv     PP.H0..H4 from coloc.abf(D1, D2)
#   coloc_abf_results.tsv     per-SNP lABF.df1/df2 and SNP.PP.H4
#   finemap_abf.tsv           finemap.abf(D1) per-SNP posterior
#   susie_lbf1.tsv, susie_lbf2.tsv   runsusie() lbf_variable matrices
#   susie_csindex.tsv         credible-set indices for each trait
#   coloc_susie_summary.tsv   per-pair PP.H0..H4 from coloc.susie
#   coloc_susie_results.tsv   per-SNP SNP.PP.H4 from coloc.susie

suppressPackageStartupMessages({
  library(coloc)
})

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[[1]] else "R_out"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

data(coloc_test_data)
D1 <- coloc_test_data$D1
D2 <- coloc_test_data$D2

# --- export the raw datasets so Python uses identical inputs ---------
write_dataset <- function(D, prefix) {
  df <- data.frame(
    snp = D$snp, position = D$position,
    beta = D$beta, varbeta = D$varbeta, MAF = D$MAF
  )
  write.table(df, file.path(out_dir, paste0(prefix, ".tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
  ld <- as.data.frame(D$LD)
  write.table(ld, file.path(out_dir, paste0(prefix, "_LD.tsv")),
              sep = "\t", quote = FALSE, row.names = TRUE,
              col.names = NA)
}
write_dataset(D1, "D1")
write_dataset(D2, "D2")

meta <- data.frame(
  trait = c("D1", "D2"),
  type = c(D1$type, D2$type),
  N = c(D1$N, D2$N),
  sdY = c(ifelse(is.null(D1$sdY), NA, D1$sdY),
          ifelse(is.null(D2$sdY), NA, D2$sdY))
)
write.table(meta, file.path(out_dir, "D_meta.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- coloc.abf ------------------------------------------------------
ca <- coloc.abf(D1, D2)
summ <- data.frame(name = names(ca$summary),
                   value = as.numeric(ca$summary))
write.table(summ, file.path(out_dir, "coloc_abf_summary.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
res <- ca$results[, c("snp", "lABF.df1", "lABF.df2",
                      "internal.sum.lABF", "SNP.PP.H4")]
write.table(res, file.path(out_dir, "coloc_abf_results.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- finemap.abf ----------------------------------------------------
fm <- finemap.abf(D1)
fm_out <- fm[, c("snp", "lABF.", "prior", "SNP.PP")]
write.table(fm_out, file.path(out_dir, "finemap_abf.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- sensitivity ----------------------------------------------------
sens <- sensitivity(ca, "H4 > 0.5", doplot = FALSE)
write.table(sens, file.path(out_dir, "sensitivity.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- case/control path: finemap.abf on a cc dataset -----------------
Dcc <- list(
  pvalues = pnorm(-abs(D1$beta / sqrt(D1$varbeta))) * 2,
  MAF = D1$MAF, N = 1000, type = "cc", s = 0.5, snp = D1$snp
)
fmcc <- finemap.abf(Dcc)
write.table(fmcc[, c("snp", "lABF.", "prior", "SNP.PP")],
            file.path(out_dir, "finemap_abf_cc.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# --- runsusie + coloc.susie ----------------------------------------
S1 <- runsusie(D1)
S2 <- runsusie(D2)

write_lbf <- function(S, prefix) {
  lbf <- as.data.frame(S$lbf_variable)
  write.table(lbf, file.path(out_dir, paste0(prefix, ".tsv")),
              sep = "\t", quote = FALSE, row.names = TRUE,
              col.names = NA)
}
write_lbf(S1, "susie_lbf1")
write_lbf(S2, "susie_lbf2")

csidx <- data.frame(
  trait = c(rep("D1", length(S1$sets$cs_index)),
            rep("D2", length(S2$sets$cs_index))),
  cs_index = c(S1$sets$cs_index, S2$sets$cs_index)
)
write.table(csidx, file.path(out_dir, "susie_csindex.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cs <- coloc.susie(S1, S2)
cs_summ <- as.data.frame(cs$summary)
write.table(cs_summ, file.path(out_dir, "coloc_susie_summary.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
if (!is.null(cs$results)) {
  cs_res <- as.data.frame(cs$results)
  write.table(cs_res, file.path(out_dir, "coloc_susie_results.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)
}

# --- coloc.signals (cond and mask) ---------------------------------
sig_cond <- coloc.signals(D1, D2, method = "cond", p12 = 1e-5)
sc <- as.data.frame(sig_cond$summary)
write.table(sc[, c("hit1", "hit2", "nsnps", "PP.H0.abf", "PP.H1.abf",
                    "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")],
            file.path(out_dir, "coloc_signals_cond.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

sig_mask <- coloc.signals(D1, D2, method = "mask", p12 = 1e-5)
sm <- as.data.frame(sig_mask$summary)
write.table(sm[, c("hit1", "hit2", "nsnps", "PP.H0.abf", "PP.H1.abf",
                    "PP.H2.abf", "PP.H3.abf", "PP.H4.abf")],
            file.path(out_dir, "coloc_signals_mask.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("R reference driver complete:", out_dir, "\n")
