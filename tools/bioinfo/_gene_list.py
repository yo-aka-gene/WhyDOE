from typing import List, Dict, Tuple
import anndata as ad
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LassoCV

from ._gene_cache_mgr import GeneCacheManager
from ._gene_query import gene_query
import tools.bioinfo.scanpy_extensions as sce


class GeneList:
    def __init__(
        self,
        adata: ad.AnnData,
        key: str,
        category: str,
        database: Dict[str, str] = None,
        gene_names: List[str] = None,
        preset: bool = False,
        source_key: str = "gene_name",
        caption: str = None,
        **kwargs
    ):
        assert (database is not None) or (gene_names is not None), \
            f"Assign at least either database or gene_names"
        self.name2idx = {n: i for n, i in zip(adata.var[source_key], adata.var.index)}
        self.idx2name = {i: n for n, i in self.name2idx.items()}
        gene_names = gene_names if gene_names is not None else database[category]
        loader = GeneCacheManager()
        self.genes = loader.load(
            key=key, 
            func=(lambda gene_names: gene_names) if preset else gene_query, 
            gene_names=gene_names, 
            source=adata.var[source_key]
        )
        self.ids = [self.name2idx[g] for g in self.genes]
        self.category = category
        self.caption = key.capitalize() if caption is None else caption
        self.score_name = f"{self.caption.replace(' ', '_')}_score"
        self.score_prob_name = self.score_name + "_prob"
        
        _data = adata.copy()

        sce.tl.prob_genes(
            _data, 
            gene_list=self.ids, 
            score_name=self.score_name, 
            copy=False,
            **kwargs
        )

        self.data = _data[:, self.ids].copy()
        


    def select_correlated_genes(
        self,
        n_top: int,
        metacell_key: str = "SEACells",
        use_raw_score: bool = False,
        **kwargs
    ) -> Tuple[List[str]]:
        score_name = self.score_name if use_raw_score else self.score_prob_name
        self.selected_ids = self.data[:, self.ids].to_df().assign(
            score=self.data.obs[score_name]
        ).assign(
            MetaCell=self.data.obs[metacell_key]
        ).groupby(
            "MetaCell", observed=False
        ).mean().corr(**kwargs)["score"].drop("score").sort_values(
            ascending=False
        ).iloc[:n_top].index.tolist()
        self.selected_genes = [
            self.idx2name[g] for g in self.selected_ids
        ]
        return (self.selected_genes, self.selected_ids)


    def select_independent_genes(
        self,
        n_top: int,
        metacell_key: str = "SEACells",
        use_raw_score: bool = False,
        n_cv: int = 5,
        step: float = 10,
        random_state: int = 0,
        **kwargs
    ) -> Tuple[List[str]]:
        score_name = self.score_name if use_raw_score else self.score_prob_name
        data = self.data[:, self.ids].to_df().assign(
            score=self.data.obs[score_name]
        ).assign(
            MetaCell=self.data.obs[metacell_key]
        ).groupby(
            "MetaCell", observed=False
        ).mean()
        X = data.loc[:, self.ids]
        y = data.loc[:, "score"].values.ravel()
        lasso_cv = LassoCV(cv=n_cv, random_state=random_state).fit(X, y)
        best_alpha = lasso_cv.alpha_
        selector = RFE(
            estimator=Lasso(alpha=best_alpha),
            n_features_to_select=n_top,
            step=step
        )
        selector.fit(X, y)
        self.selected_ids = [
            g for g, s in zip(self.ids, selector.support_) if s
        ]
        self.selected_genes = [
            self.idx2name[g] for g in self.selected_ids
        ]
        return (self.selected_genes, self.selected_ids)


    def get_matrix(
        self,
        use_mean_per_metacell: bool = True,
        metacell_key: str = "SEACells",
        with_score: bool = False,
        use_raw_score: bool = False,
        use_gene_name: bool = True
    ) -> pd.DataFrame:
        data = pd.DataFrame(
            self.data[:, self.selected_ids].X,
            index=self.data.obs_names,
            columns=self.selected_genes if use_gene_name else self.selected_ids
        )
        score_name = self.score_name if use_raw_score else self.score_prob_name
        data = data.assign(
            score=self.data.obs[score_name]
        ) if with_score else data
        data = data.assign(
            MetaCell=self.data.obs[metacell_key]
        ).groupby(
            "MetaCell", observed=False
        ).mean() if use_mean_per_metacell else data
        return data