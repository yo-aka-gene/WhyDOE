import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import dunnett

from doe_modules.simulation import AbstractSimulator
from doe_modules.preferences.pvalues import sign, select, asterisk, p_format


def legacy_scatterview(
    simulation: AbstractSimulator,
    cmap: list = None,
    const_color: tuple = plt.cm.gray(.2),
    ax: plt.Axes = None,
):
    assert issubclass(type(simulation), AbstractSimulator), \
        f"pass subclass of AbstractSimulator, got {simulation}[{type(simulation)}]"
    assert simulation.is_executed, \
        f"Simluation is not excecuted yet. Run simulation.simulate before passing it to legacy_scatterview"
    assert simulation.metadata['design'] == "C+LOO", \
        f"Only simulation models based on the C+LOO design is supported; got {simulation.metadata['design']}"

    ax = plt.subplots()[1] if ax is None else ax
    nrows = simulation.n_factor + 1

    sns.scatterplot(
        y=["all factors" if i == 0 else f"X{i} KD" for i in range(nrows)] * simulation.metadata['n_rep'], 
        x=simulation.exresult, s=30, color = ([const_color] + cmap) * simulation.metadata['n_rep']
    )

    counterpart = [None] + [
        np.mean([v for loc, v in enumerate(simulation.exresult) if loc % nrows != i]) for i in range(simulation.n_factor)
    ]

    sns.scatterplot(
        y=["control" if i == 0 else f"X{i} KD" for i in range(nrows)], 
        x=counterpart, s=30, marker="x", color=".2", label="control"
    )

    ax.set(
        xlabel="output values", ylabel="", 
        title=f"{simulation.metadata['design']} design (N={simulation.metadata['n_rep']})"
    )


def bio_scatterview(
    simulation: AbstractSimulator,
    cmap: list = None,
    alpha: float = .7,
    const_color: tuple = plt.cm.gray(.2),
    ax: plt.Axes = None
):
    assert issubclass(type(simulation), AbstractSimulator), \
        f"pass subclass of AbstractSimulator, got {simulation}[{type(simulation)}]"
    assert simulation.is_executed, \
        f"Simluation is not excecuted yet. Run simulation.simulate before passing it to legacy_scatterview"
    assert simulation.metadata['design'] == "C+LOO", \
        f"Only simulation models based on the C+LOO design is supported; got {simulation.metadata['design']}"

    ax = plt.subplots()[1] if ax is None else ax
    nrows = simulation.n_factor + 1

    sns.scatterplot(
        y=["all factors" if i == 0 else f"X{i} KD" for i in range(nrows)] * simulation.metadata['n_rep'], 
        x=simulation.exresult, s=30, color = ([const_color] + cmap) * simulation.metadata['n_rep'],
        alpha=alpha, ax=ax
    )

    sns.scatterplot(
        y=["all factors" if i == 0 else f"X{i} KD" for i in range(nrows)],
        x=simulation.exresult.reshape(3, -1).mean(axis=0),
        s=30, marker=",", color = [const_color] + cmap, 
        label="mean values", zorder=-100, ax=ax
    )

    ylim = ax.get_ylim()
    ax.vlines(
        simulation.exresult.reshape(3, -1).mean(axis=0)[0], *ylim, 
        color=".2", linewidth=1
    )
    ax.set_ylim(ylim)
    ax.set(
        xlabel="output values", ylabel="", 
        title=f"{simulation.metadata['design']} design (N={simulation.metadata['n_rep']})"
    )


def bio_multicomp(
    simulation: AbstractSimulator,
    test_kwargs: dict = {"alternative": "two-sided", "random_state": 0},
    text_kwargs: dict = {},
    cmap: list = None,
    alpha: float = 1,
    const_color: tuple = plt.cm.gray(.2),
    ax: plt.Axes = None,
    xscales: np.ndarray = np.array([1.3, 1.8]),
    jitter_ratio: float = .2,
    display_pvals: bool = False
):
    assert issubclass(type(simulation), AbstractSimulator), \
        f"pass subclass of AbstractSimulator, got {simulation}[{type(simulation)}]"
    assert simulation.is_executed, \
        f"Simluation is not excecuted yet. Run simulation.simulate before passing it to legacy_scatterview"
    assert simulation.metadata['design'] == "C+LOO", \
        f"Only simulation models based on the C+LOO design is supported; got {simulation.metadata['design']}"

    ax = plt.subplots()[1] if ax is None else ax
    nrows = simulation.n_factor + 1

    df = pd.DataFrame({
        "group": [
            "all factors" if i == 0 else f"X{i} KD" for i in range(nrows)
        ] * simulation.metadata['n_rep'],
        "y": simulation.exresult
    })
    pvals = dunnett(
        *[df[df.group == g].y for g in df[df.group != "all factors"].group.unique()],
        control=df[df.group == "all factors"].y,
        **test_kwargs
    ).pvalue

    sns.stripplot(
        y=["all factors" if i == 0 else f"X{i} KD" for i in range(nrows)] * simulation.metadata['n_rep'], 
        x=simulation.exresult, s=5, 
        hue=([const_color] + cmap) * simulation.metadata['n_rep'],
        palette=[const_color] + cmap, 
        legend=False,
        alpha=1, ax=ax,
        # linewidth=.5, edgecolor=".2",
        jitter=.2
    )
    
    means = simulation.exresult.reshape(-1, nrows).T.mean(axis=1)
    sds = simulation.exresult.reshape(-1, nrows).T.std(axis=1)

    xm, xM = ax.get_xlim()
    jitter = jitter_ratio * (xM - xm)
    
    yrange = (lambda v1, v2: v2 - v1)(*np.sort(np.array(ax.get_ylim())))
    
    for i, m, sd, c in zip(range(nrows), means, sds, [const_color] + cmap):
        ax.vlines(m - sd, i - (yrange * .03), i + (yrange * .03), color=c, linewidth=1.5)
        ax.vlines(m + sd, i - (yrange * .03), i + (yrange * .03), color=c, linewidth=1.5)
        ax.vlines(m, i - (yrange * .02), i + (yrange * .02), color=c, linewidth=1.5)
        ax.hlines(i, m - sd, m + sd, color=c, linewidth=1.5)
    
    default_text_kwargs = dict(color=".2", size=6, ha="center", va="center")

    for i, name in enumerate(df[df.group != "all factors"].group.unique()):
        c = select(df[df.group == name].y)
        p = pvals[i]
        ax.text(
            sign(c) * (abs(c) + jitter), i + 1, 
            asterisk(p) + p_format(p) if display_pvals else asterisk(p), 
            **{**default_text_kwargs, **text_kwargs}
        )

    ax.set_xlim(xscales * np.array([*ax.get_xlim()]))

    ax.set(
        xlabel="output values", ylabel="", 
        title=f"{simulation.name} analyzed with {simulation.metadata['design']} design (N={simulation.metadata['n_rep']})"
    )