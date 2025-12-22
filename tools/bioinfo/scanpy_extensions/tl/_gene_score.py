from typing import Union

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.stats import zscore


def prob_genes(
    adata: ad.AnnData,
    gene_list: list,
    **kwargs
) -> Union[None, ad.AnnData]:
    score_name = kwargs.get("score_name", "score")
    result = sc.tl.score_genes(
        adata=adata, 
        gene_list=gene_list,
        **kwargs
    )

    sigmoid = lambda s: 1 / (1 + np.exp(-zscore(s, nan_policy="omit")))

    data = result if result is not None else adata
    data.obs[f"{score_name}_prob"] = sigmoid(data.obs[score_name])

    return result


def score_genes_cell_cycle(
    adata: ad.AnnData,
    s_genes: list,
    g2m_genes: list,
    **kwargs
) -> None:
    prob_genes(
        adata=adata, 
        gene_list=s_genes, 
        score_name='S_score', 
        copy=False,
        **kwargs
    )
    prob_genes(
        adata=adata, 
        gene_list=g2m_genes, 
        score_name='G2M_score', 
        copy=False,
        **kwargs
    )
    sc.tl.score_genes_cell_cycle(
        adata=adata,
        s_genes=s_genes, 
        g2m_genes=g2m_genes,
        copy=False,
        **kwargs
    )
    adata.obs["phase"] = adata.obs["phase"].map(lambda p: p if p != "G2M" else "G2/M")
