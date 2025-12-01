from ._corrcoeff import correlation_heatmap
from ._design_matrix import design_heatmap
from ._cloo_plots import bio_scatterview, bio_multicomp
from ._splat import generate_clusters


__all__ = [
    correlation_heatmap,
    design_heatmap,
    bio_scatterview,
    bio_multicomp,
    generate_clusters
]