### d_optimization.R ###

options(warn = -1)

# packages <- c("AlgDesign")

# for (pkg in packages) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     install.packages(pkg, repos="https://cloud.r-project.org/")
#   }
#   suppressPackageStartupMessages(library(pkg, character.only = TRUE))
# }


# optslist <- list(
#   make_option(
#     c("-t", "--tempdir"),
#     typ = "character",
#     default = "/home/jovyan/out",
#     help = "temporary directory to save intermediate files"
#   ),
#   make_option(
#     c("-a", "--add"),
#     typ = "integer",
#     default = 0,
#     help = "number of trials to add to the original design matrix"
#   )
# )
# parser <- OptionParser(option_list = optslist)
# opts <- parse_args(parser)

# arr <- read_feather(paste0(opts$tempdir, "/arr.feather"), mmap = FALSE)
# candidate <- read_feather(paste0(opts$tempdir, "/candidate.feather"))
# n_add <- opts$add

d_optimize_core <- function(dsmatrix, candidate, n_add, random_state) {
    set.seed(random_state)
    
    optimized <- optFederov(
        ~., 
        data = rbind(dsmatrix, candidate), 
        nTrials=nrow(dsmatrix) + n_add, 
        criterion = "D", 
        augment = TRUE, 
        rows = 1:nrow(dsmatrix), 
        maxIteration = choose(nrow(candidate), n_add)
    )
    return(optimized$design)
}

options(warn = 0)