### dunnett_power.R ###
# print(commandArgs(trailingOnly = TRUE))

options(warn = -1)

packages <- c("optparse", "arrow", "mvtnorm", "stats", "yaml")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos="https://cloud.r-project.org/")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

optslist <- list(
  make_option(
    c("-t", "--tempdir"),
    type = "character",
    default = "/home/jovyan/out",
    help = "temporary directory to save intermediate files"
  ),
  make_option(
    c("-m", "--m0"),
    type = "double",
    help = "mean of the control group"
  ),
  make_option(
    c("-n", "--n_rep"),
    type = "integer",
    help = "number of replication"
  ),
  make_option(
    c("-s", "--sigma"),
    type = "double",
    help = "estimated sd (shared)"
  ),
  make_option(
    c("-a", "--alpha"),
    type = "double",
    help = "alpha"
  )
)
parser <- OptionParser(option_list = optslist)
opts <- parse_args(parser)

m0 <- opts$m0
m_i <- read_feather(paste0(opts$tempdir, "/arr.feather"), mmap = FALSE)$column_0
n_rep <- opts$n_rep
sigma <- opts$sigma
alpha <- opts$alpha


dunnett_power_analytic <- function(
  mu0, mu_t, n0, nt, sigma, alpha
){
  k  <- length(mu_t)
  ni <- rep(nt, k)
  se <- sqrt(1/n0 + 1/ni) * sigma
  ncp <- (mu_t - mu0)/se

  #  df = N_total - G,  G = k + 1
  df  <- n0 + sum(ni) - (k + 1)

  # corr((Ti-C),(Tj-C)) = (1/n0) / sqrt((1/ni + 1/n0)(1/nj + 1/n0))
  R <- diag(k)
  for(i in 1:k){
    for(j in 1:k){
      if(i != j){
        R[i,j] <- (1/n0) / sqrt((1/ni[i] + 1/n0)*(1/ni[j] + 1/n0))
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

power <- dunnett_power_analytic(
    mu0 = m0, mu_t = m_i, n0 = n_rep, nt = n_rep, sigma = sigma, alpha = alpha
)

# write_feather(
#   power,
#   paste0(opts$tempdir, "/power.feather")
# )

write_yaml(
  power,
  paste0(opts$tempdir, "/power.yaml")
)

options(warn = 0)