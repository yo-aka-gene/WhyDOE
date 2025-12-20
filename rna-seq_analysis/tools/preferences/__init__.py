import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path


mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


kwarg_savefig = {
    "facecolor": "white",
    "dpi": 600,
    "bbox_inches": "tight",
    "pad_inches": 0.05,
    "transparent": True,
}

kwarg_save_transparent_fig = {
    "dpi": 600,
    "bbox_inches": "tight",
    "pad_inches": 0.05,
    "transparent": True,
}


outputdir = Path(__file__).resolve().parent.parent.parent / "out"


def level_cmap(
    color: tuple = tuple(
        (3 * np.array(plt.cm.Purples(.6)) + 2.5 * np.array([0, 0, 1, 1])) / 5
    ), 
    intensity: float = .6, 
    grayscale: float = .2,
    gamma: float = 2,
    name: str = None
):   
    cdict = {
        "red": [
            (x, y0, y1) for x, y0, y1 in zip(
                np.linspace(0, 1, 256),
                np.linspace(plt.cm.Greys(grayscale)[0], color[0], 256),
                np.linspace(plt.cm.Greys(grayscale)[0], color[0], 256),  
            )
        ],
        "green": [
            (x, y0, y1) for x, y0, y1 in zip(
                np.linspace(0, 1, 256),
                np.linspace(plt.cm.Greys(grayscale)[1], color[1], 256),
                np.linspace(plt.cm.Greys(grayscale)[1], color[1], 256),  
            )
        ],
        "blue": [
            (x, y0, y1) for x, y0, y1 in zip(
                np.linspace(0, 1, 256),
                np.linspace(plt.cm.Greys(grayscale)[2], color[2], 256),
                np.linspace(plt.cm.Greys(grayscale)[2], color[2], 256),  
            )
        ],
    }

    name = name if name is not None else f"level_cmap_{color}"

    return LinearSegmentedColormap(name, cdict, gamma=gamma)


seurat = level_cmap(name="seurat")


rgba2gray = lambda r, g, b, a: a * (0.299 * r + 0.578 * g + 0.114 * b)


def textcolor(c, thresh=.4, dark=".2", light="1") -> str:
    if isinstance(c, str):
        return dark if float(c) > thresh else light
    else:
        return dark if rgba2gray(*c) > thresh else light
