from . import cmap
from . import pvalues
from ._harmonic_mean import harmonic_mean
from ._subplots import subplots
from ._textcolor import rgba2gray, textcolor

import numpy as np


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


kwarg_bootstrap = dict(
    statistic=np.mean, 
    n_resamples=10000, 
    confidence_level=0.95,
    random_state=np.random.default_rng()
)


kwarg_err = dict(capsize=.25, err_kws={'linewidth': 1}, linewidth=1,)


def order2_interaction_regex(key: int, id_min: int, id_max: int) -> str:
    regex = f"^X[{key}]$"
    regex += f"|^X{id_min}X[{id_min + 1}-{id_max}]$" if key == id_min \
        else f"|^X[{id_min}-{key - 1}]X{key}$|^X{key}X[{id_min}-{id_max}]$"
    return regex


def f2s(z: float, digit: int = 3) -> str:
    s = str(round(z, digit))
    subzero_digit = len(s.split(".")[-1])
    return s if subzero_digit == digit else s + "".join(["0" for _ in range(digit - subzero_digit)])


def fmt_suffix(suffix: str) -> str:
    if (suffix == "") or (suffix[0] == "_"):
        return suffix
    else:
        return "_" + suffix



__all__ = [
    kwarg_savefig,
    outputdir,
    heatmap_pref,
    dsmat_pref,
    kwarg_bootstrap,
    kwarg_err,
    order2_interaction_regex,
    f2s,
    fmt_suffix,
    cmap,
    pvalues,
    harmonic_mean,
    rgba2gray,
    subplots,
    textcolor,
]