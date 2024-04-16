import matplotlib.pyplot as plt
import seaborn as sns

from code.design import DOE


def correlation_heatmap(
    design: DOE,
    n_factor: int,
    cbar_kws: dict = {"label": r"$|\rho|$"},
    ax: plt.Axes = None,
    **kwargs
) -> None:
    assert issubclass(design, DOE), \
        f"pass subclass of DOE, got {design}[{type(design)}]"
    assert design().is_initialized, \
        "pass DOE without initialization"
    assert isinstance(cbar_kws, dict), \
        f"pass dict, got {cbar_kws}[{type(cbar_kws)}]"
    assert "label" in cbar_kws, \
        f"pass dict that has 'label' as a key, got {cbar_kws}"
    _, ax = plt.subplots() if ax is None else (None, ax)

    sns.heatmap(
        design().get_exmatrix(n_factor).interactions().corr().abs(),
        vmax=1, vmin=0, 
        cbar_kws={"label": r"$|\rho|$"},
        ax=ax,
        **kwargs
    )