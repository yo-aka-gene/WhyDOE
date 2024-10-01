from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt


def subplots(
    nrows: int = None,
    ncols: int = None,
    n: int = None,
    base: tuple = None,
    figsize: tuple = None,
    auto: bool = True,
    vertical: bool = True,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    if (n is None) and (nrows is None) and (ncols is None):
        nrows, ncols = 1, 1
    elif (n is None) and isinstance(nrows, int) and (ncols is None) and auto:
        n = nrows
        nrows = None
    elif (n is None) and isinstance(nrows, int) and (ncols is None) and not auto:
        nrows, ncols = nrows, 1
    elif (n is None) and (nrows is None) and isinstance(ncols, int):
        nrows, ncols = 1, ncols
    elif (n is None) and isinstance(nrows, int) and isinstance(ncols, int) and auto:
        n = nrows
        nrows, ncols = None, ncols
    elif (n is None) and isinstance(nrows, int) and isinstance(ncols, int) and not auto:
        pass

    if isinstance(n, int) and (nrows is None) and (ncols is None):
        Num, num = np.ceil(np.sqrt(n)),  np.ceil(n / np.ceil(np.sqrt(n)))
        nrows = Num if vertical else num
        ncols = num if vertical else Num

    elif isinstance(n, int) and isinstance(nrows, int) and (ncols is None):
        nrows, ncols = nrows, np.ceil(n / nrows)

    elif isinstance(n, int) and (nrows is None) and isinstance(ncols, int):
        nrows, ncols = np.ceil(n / ncols), ncols

    if isinstance(base, tuple):
        figsize = (base[0] * ncols, base[1] * nrows)

    return plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwarg)
        