import anndata as ad
import numpy as np


def get_quantiles(
    data: ad.AnnData, 
    metrics: str, 
    by: float = .1,
    area: list = None
):
    area = np.arange(0, 1, by)[1:] if area is None else area
    return [
        data.obs[metrics].quantile(v) for v in area
    ]