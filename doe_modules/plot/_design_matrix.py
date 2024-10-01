import matplotlib.pyplot as plt
import seaborn as sns

from doe_modules.design import DOE


def design_heatmap(
    design: DOE,
    n_factor: int,
    annot_kws: dict = {"+": "+", "-": "$-$"},
    design_kws: dict = {},
    ax: plt.Axes = None,
    title: str = None,
    **kwargs
) -> None:
    assert issubclass(design, DOE), \
        f"pass subclass of DOE, got {design}[{type(design)}]"
    assert isinstance(annot_kws, dict), \
        f"pass dict, got {annot_kws}[{type(annot_kws)}]"
    assert "+" in annot_kws and "-" in annot_kws, \
        f"pass dict that has '+' and '-' as keys, got {annot_kws}"
    _, ax = plt.subplots() if ax is None else (None, ax)

    design = design()
    dsm = design.get_exmatrix(n_factor, **design_kws)(encode=True)

    sns.heatmap(
        dsm,
        annot=dsm.applymap(
            lambda x: annot_kws["+"] if x else annot_kws["-"]
        ),
        fmt="s", cbar=False, ax=ax,
        **kwargs
    )
    title = design.title if title is None else title
    ax.set(title=title)
