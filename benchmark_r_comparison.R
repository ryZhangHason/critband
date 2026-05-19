# critband vs R Benchmark Script
# Compares critband critical_bandwidth, dip_test, find_modes with R equivalents

library(multimode)
library(diptest)
library(ks)

set.seed(42)

# ---- Benchmark data generators (matches critband/benchmark.py) ----

generate_mixture <- function(components, seed) {
  set.seed(seed)
  result <- c()
  for (i in seq_len(nrow(components))) {
    comp <- components[i, ]
    result <- c(result, rnorm(comp[3], comp[1], comp[2]))
  }
  return(result)
}

benchmark_cases <- list(
  well_separated_equal_var = list(
    gen = function(s) generate_mixture(rbind(c(-2,0.3,200), c(2,0.3,200)), s),
    desc = "N(-2,0.3) ∪ N(2,0.3), n=400",
    pola_h = 1.8631
  ),
  moderate_separation = list(
    gen = function(s) generate_mixture(rbind(c(-1,0.5,250), c(1.5,0.5,250)), s+1),
    desc = "N(-1,0.5) ∪ N(1.5,0.5), n=500",
    pola_h = 1.0953
  ),
  barely_separated = list(
    gen = function(s) generate_mixture(rbind(c(-0.5,0.4,300), c(0.5,0.4,300)), s+2),
    desc = "N(-0.5,0.4) ∪ N(0.5,0.4), n=600",
    pola_h = 0.2788
  ),
  unequal_variance = list(
    gen = function(s) generate_mixture(rbind(c(-2,0.6,200), c(2,0.2,200)), s+3),
    desc = "N(-2,0.6) ∪ N(2,0.2), n=400",
    pola_h = 1.7831
  ),
  unequal_weights = list(
    gen = function(s) generate_mixture(rbind(c(-2,0.3,100), c(2,0.3,400)), s+4),
    desc = "N(-2,0.3,100) ∪ N(2,0.3,400), n=500",
    pola_h = 1.2592
  ),
  extreme_separation = list(
    gen = function(s) generate_mixture(rbind(c(-5,0.5,200), c(5,0.5,200)), s+5),
    desc = "N(-5,0.5) ∪ N(5,0.5), n=400",
    pola_h = 4.6940
  ),
  trimodal = list(
    gen = function(s) generate_mixture(rbind(c(-3,0.3,150), c(0,0.3,150), c(3,0.3,150)), s+6),
    desc = "3× N(±3,0.3,150), n=450",
    pola_h = 1.3810
  ),
  skewed_bimodal = list(
    gen = function(s) generate_mixture(rbind(c(-1.5,0.4,350), c(2.0,0.6,150)), s+7),
    desc = "N(-1.5,0.4,350) ∪ N(2.0,0.6,150)",
    pola_h = 1.1416
  ),
  heavy_tailed_bimodal = list(
    gen = function(s) generate_mixture(rbind(c(-3,0.8,200), c(3,0.8,200)), s+8),
    desc = "N(-3,0.8) ∪ N(3,0.8), n=400",
    pola_h = 2.7082
  ),
  near_unimodal = list(
    gen = function(s) generate_mixture(rbind(c(0,0.6,300), c(1.5,0.6,300)), s+9),
    desc = "N(0,0.6,300) ∪ N(1.5,0.6,300)",
    pola_h = 0.4182
  ),
  small_sample_bimodal = list(
    gen = function(s) generate_mixture(rbind(c(-2,0.5,30), c(2,0.5,30)), s+10),
    desc = "N(-2,0.5,30) ∪ N(2,0.5,30), n=60",
    pola_h = 1.8608
  ),
  overlapping_variances = list(
    gen = function(s) generate_mixture(rbind(c(-0.8,0.7,250), c(0.8,0.5,250)), s+11),
    desc = "N(-0.8,0.7,250) ∪ N(0.8,0.5,250)",
    pola_h = 0.4593
  )
)

# ---- Helper: bootstrap-mode p-value from modetest ----
# For each case, extract the p-value and h_crit equivalent from modetest

run_modetest <- function(x) {
  t <- system.time({
    result <- modetest(x, mod0=1, method="ACR", B=199)
  })
  list(
    p_value = result$p.value,
    statistic = result$statistic,
    time = t["elapsed"]
  )
}

# ---- Helper: dip.test ----
run_diptest <- function(x) {
  t <- system.time({
    result <- dip.test(x)
  })
  list(
    dip = result$statistic,
    p_value = result$p.value,
    time = t["elapsed"]
  )
}

# ---- Helper: nmodes estimation ----
run_nmodes <- function(x) {
  # nmodes now requires bw; use Silverman's rule (nrd0) — matches critband's default
  bw <- bw.nrd0(x)
  t <- system.time({
    result <- tryCatch(nmodes(x, bw=bw), error=function(e) NA_integer_)
  })
  if (is.na(result)) {
    list(n_modes = NA, time = NA)
  } else {
    list(n_modes = result, time = t["elapsed"])
  }
}

# ---- Main benchmark loop ----
cat("case,desc,n,pola_h_crit,modetest_p,modetest_stat,modetest_time,")
cat("diptest_dip,diptest_p,diptest_time,")
cat("nmodes_n,nmodes_time\n")

for (name in names(benchmark_cases)) {
  case <- benchmark_cases[[name]]
  x <- case$gen(42)
  n <- length(x)
  
  cat(name, ",", sep="")
  cat(gsub(",", ";", case$desc), ",", sep="")
  cat(n, ",", sep="")
  cat(case$pola_h, ",", sep="")
  
  # modetest (Silverman)
  mt <- tryCatch(run_modetest(x), error=function(e) list(p_value=NA, statistic=NA, time=NA))
  cat(mt$p_value, ",", mt$statistic, ",", mt$time, ",", sep="")
  
  # dip.test (Hartigan)
  dt <- tryCatch(run_diptest(x), error=function(e) list(dip=NA, p_value=NA, time=NA))
  cat(dt$dip, ",", dt$p_value, ",", dt$time, ",", sep="")
  
  # nmodes
  nm <- tryCatch(run_nmodes(x), error=function(e) list(n_modes=NA, time=NA))
  cat(nm$n_modes, ",", nm$time, "\n", sep="")
}

cat("\n--- DONE ---\n")
