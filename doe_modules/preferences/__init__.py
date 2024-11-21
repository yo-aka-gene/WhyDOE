from . import cmap
from . import pvalues
from ._subplots import subplots
from ._textcolor import rgba2gray, textcolor


kwarg_savefig = {
    "facecolor": "white",
    "dpi": 300,
    "bbox_inches": "tight",
    "pad_inches": 0.05,
    "transparent": True,
}

outputdir = "/home/jovyan/out"

heatmap_pref = dict(
    vmax=1, vmin=0, 
    cbar_kws={"label": r"$|\rho|$"},
    cmap="coolwarm", square=True,
)

dsmat_pref = dict(
    cmap="Purples", square=True
)


__all__ = [
    kwarg_savefig,
    outputdir,
    heatmap_pref,
    dsmat_pref,
    cmap,
    pvalues,
    rgba2gray,
    subplots,
    textcolor,
]