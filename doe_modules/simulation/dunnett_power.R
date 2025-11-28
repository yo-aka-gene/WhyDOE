### dunnett_power.R ###

options(warn = -1)

# packages <- c("mvtnorm", "stats")

# for (pkg in packages) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     install.packages(pkg, repos="https://cloud.r-project.org/")
#   }
#   suppressPackageStartupMessages(library(pkg, character.only = TRUE))
# }

dunnett_power_analytic <- function(mu0, mu_t, n0, nt, sigma, alpha) {
  k  <- length(mu_t)
  ni <- rep(nt, k)
  se <- sqrt(1 / n0 + 1 / ni) * sigma
  ncp <- as.numeric((mu_t - mu0) / se)

  #  df = N_total - G,  G = k + 1
  df  <- n0 + sum(ni) - (k + 1)

  # corr((Ti-C),(Tj-C)) = (1 / n0) / sqrt((1 / ni + 1 / n0)(1 / nj + 1 / n0))
  R <- diag(k)
  for(i in 1:k){
    for(j in 1:k){
      if(i != j){
        R[i,j] <- (1 / n0) / sqrt((1 / ni[i] + 1 / n0) * (1 / ni[j] + 1 / n0))
      }
    }
  }

  # P(max |T_i| <= c) = 1 - alpha
  q <- qmvt(p = 1 - alpha, tail = "both.tails", df = df, corr = R)$quantile
  crit <- as.numeric(q)

  # per-comparison power i： 1 - [F(c; df, ncp) - F(-c; df, ncp)]
  pc_power <- 1 - (pt(crit, df, ncp) - pt(-crit, df, ncp))

  # familywise power：1 - P(\forall |T_i| <= c)
  keep_prob <- pmvt(lower = rep(-crit, k), upper = rep(crit, k),
                    df = df, corr = R, delta = ncp)
  fw_power <- 1 - as.numeric(keep_prob)

  list(
    df = df, crit = crit, 
    # R = R,
    per_comparison_power = pc_power,
    familywise_power = fw_power
  )
}

options(warn = 0)