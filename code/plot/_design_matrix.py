import matplotlib.pyplot as plt
import seaborn as sns

from code.design import DOE


def design_heatmap(
    design: DOE,
    n_factor: int,
    annot_kws: dict = {"+": "+", "-": "-"},
    ax: plt.Axes = None,
    **kwargs
) -> None:
    assert issubclass(design, DOE), \
        f"pass subclass of DOE, got {design}[{type(design)}]"
    assert design().is_initialized, \
        "pass DOE without initialization"
    assert isinstance(annot_kws, dict), \
        f"pass dict, got {annot_kws}[{type(annot_kws)}]"
    assert "+" in annot_kws and "-" in annot_kws, \
        f"pass dict that has '+' and '-' as keys, got {annot_kws}"
    _, ax = plt.subplots() if ax is None else (None, ax)

    dsm = design().get_exmatrix(n_factor)(binarize=True)

    sns.heatmap(
        dsm,
        annot=dsm.applymap(
            lambda x: annot_kws["+"] if x else annot_kws["-"]
        ),
        fmt="s", cbar=False, ax=ax,
        **kwargs
    )