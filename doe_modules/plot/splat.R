### splat.R ###

options(warn = -1)

# packages <- c("BiocManager")

# for (pkg in packages) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     install.packages(pkg, repos="https://cloud.r-project.org/")
#   }
#   suppressPackageStartupMessages(library(pkg, character.only = TRUE))
# }

# bio_packages <- c("splatter", "scater")

# for (pkg in bio_packages) {
#   if (!requireNamespace(pkg, quietly = TRUE)) {
#     BiocManager::install(pkg)
#   }
#   suppressPackageStartupMessages(library(pkg, character.only = TRUE))
# }

generate_clusters <- function(n_genes, n_cells, group_prob, de_prob, dropout_mid, random_state) {
    set.seed(random_state)
    params <- newSplatParams()
    params <- setParams(params, 
        nGenes = n_genes, 
        batchCells = n_cells, 
        group.prob = as.numeric(group_prob),
        de.prob = de_prob,
        dropout.type = "experiment",
        dropout.mid = dropout_mid, 
        dropout.shape = -1
    )
    
    sim <- splatSimulate(params, method = "groups", verbose = FALSE)
    list(
        counts = as.matrix(counts(sim)), 
        meta = as.data.frame(colData(sim))
    )
}

options(warn = 0)