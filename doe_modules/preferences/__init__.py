from . import cmap
from . import pvalues
from ._harmonic_mean import harmonic_mean
from ._subplots import subplots
from ._textcolor import rgba2gray, textcolor


kwarg_savefig = {
    "facecolor": "white",
    "dpi": 600,
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


def order2_interaction_regex(key: int, id_min: int, id_max: int) -> str:
    regex = f"^X[{key}]$"
    regex += f"|^X{id_min}X[{id_min + 1}-{id_max}]$" if key == id_min \
        else f"|^X[{id_min}-{key - 1}]X{key}$|^X{key}X[{id_min}-{id_max}]$"
    return regex


__all__ = [
    kwarg_savefig,
    outputdir,
    heatmap_pref,
    dsmat_pref,
    order2_interaction_regex,
    cmap,
    pvalues,
    harmonic_mean,
    rgba2gray,
    subplots,
    textcolor,
]