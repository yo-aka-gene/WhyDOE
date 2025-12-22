import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


sc.set_figure_params(scanpy=False, vector_friendly=True, dpi_save=600)


def umap(adata, title: str = None, **kwargs):
    ax = kwargs.get("ax", None)
    create_new_ax = True if ax is None else False
    if ax is None:
        fig, ax = plt.subplots()

    color = kwargs.get("color", None)     

    kwargs = {**kwargs, **dict(ax=ax, show=False, colorbar_loc=None)}
    
    sc.pl.umap(adata, **kwargs)

    if (color is not None) and isinstance(adata.obs[color].dtype, pd.CategoricalDtype): 
        ax.legend(
            loc="center left", bbox_to_anchor=(1, .5), 
            title=color.capitalize() if title is None else title, 
            frameon=False
        )

    elif (color is not None) and not isinstance(adata.obs[color].dtype, pd.CategoricalDtype): 
        mappable = ax.collections[0]
        cax = ax.inset_axes([1.02, 0.1, 0.04, 0.3])
        cbar = plt.colorbar(mappable, cax=cax)
        cbar.set_label(
            color.capitalize() if title is None else title, 
            rotation=90, labelpad=5
        )
        cbar.outline.set_visible(False)


    ax.axis("off")
    ax.set_aspect('equal')
    ax.set(title="");
    return (fig, ax) if create_new_ax else None
