import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from doe_modules.simulation import AbstractSimulator


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
        alpha=alpha
    )

    sns.scatterplot(
        y=["all factors" if i == 0 else f"X{i} KD" for i in range(nrows)],
        x=simulation.exresult.reshape(3, -1).mean(axis=0),
        s=30, marker=",", color = [const_color] + cmap, 
        label="mean values", zorder=-100
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
