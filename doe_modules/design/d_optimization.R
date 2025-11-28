### d_optimization.R ###

options(warn = -1)

# packages <- c("AlgDesign")

# for (pkg in packages) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     install.packages(pkg, repos="https://cloud.r-project.org/")
#   }
#   suppressPackageStartupMessages(library(pkg, character.only = TRUE))
# }

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