import matplotlib.pyplot as plt
import seaborn as sns

from doe_modules.design import DOE


def correlation_heatmap(
    design: DOE,
    n_factor: int,
    cbar_kws: dict = {"label": r"$|\rho|$"},
    design_kws: dict = {},
    ax: plt.Axes = None,
    title: str = None,
    **kwargs
) -> None:
    assert issubclass(design, DOE), \
        f"pass subclass of DOE, got {design}[{type(design)}]"
    assert isinstance(cbar_kws, dict), \
        f"pass dict, got {cbar_kws}[{type(cbar_kws)}]"
    assert "label" in cbar_kws, \
        f"pass dict that has 'label' as a key, got {cbar_kws}"
    _, ax = plt.subplots() if ax is None else (None, ax)

    for k, v in dict(vmax=1, vmin=0, cbar_kws={"label": r"$|\rho|$"}).items():
        kwargs = kwargs if k in kwargs else {**kwargs, k: v}

    design = design()

    sns.heatmap(
        design.get_exmatrix(n_factor, **design_kws).interactions().corr().abs(),
        ax=ax,
        **kwargs
    )
    title = design.title if title is None else title
    ax.set(title=title)
