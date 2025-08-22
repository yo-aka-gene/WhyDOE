### d_optimization.R ###

options(warn = -1)

packages <- c("optparse", "arrow", "AlgDesign")

for (pkg in packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos="https://cloud.r-project.org/")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

optslist <- list(
  make_option(
    c("-t", "--tempdir"),
    typ = "character",
    default = "/home/jovyan/out",
    help = "temporary directory to save intermediate files"
  ),
  make_option(
    c("-a", "--add"),
    typ = "integer",
    default = 0,
    help = "number of trials to add to the original design matrix"
  )
)
parser <- OptionParser(option_list = optslist)
opts <- parse_args(parser)

arr <- read_feather(paste0(opts$tempdir, "/arr.feather"), mmap = FALSE)
candidate <- read_feather(paste0(opts$tempdir, "/candidate.feather"))
n_add <- opts$add

opt <- optFederov(
  ~., rbind(arr, candidate), 
  nTrials=nrow(arr) + opts$add, 
  criterion = "D", augment = TRUE, 
  rows = 1:nrow(arr), 
  maxIteration = choose(nrow(candidate), opts$add)
)

write_feather(
  opt$design,
  paste0(opts$tempdir, "/opt.feather")
)

options(warn = 0)