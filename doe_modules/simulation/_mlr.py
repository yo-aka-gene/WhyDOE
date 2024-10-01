import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from ._abstract import AbstractSimulator
from doe_modules.design import DesignMatrix


sign = lambda x: 2 * int(x >= 0) - 1

def asterisk(p: float) -> str:
    if p >= .05:
        return "N.S."
    elif .01 <= p < .05:
        return "*"
    elif .001 <= p < .01:
        return "**"
    else:
        return "***"

    
def p_format(p: float, digit: int = 3) -> str:
    p_str = f"p={round(p, digit)}" if p >= .001 else "p<0.001"
    if len(p_str) != digit + len("p=0."):
        p_str = (p_str + "0" * (digit - 1))[:digit + len("p=0.")]
    return "\n(" + p_str + ")"


class MLR:
    def __init__(
        self,
        simulation: AbstractSimulator,
        interactions: bool = False
    ):
        assert issubclass(type(simulation), AbstractSimulator), \
            f"pass subclass of AbstractSimulator, got {simulation}[{type(simulation)}]"
        assert simulation.is_executed, \
            f"Simluation is not excecuted yet. Run simulation.simulate before passing it to MLR.__init__"
        exog = DesignMatrix(simulation.exmatrix.values).interactions() if interactions else simulation.exmatrix
        self.result = sm.OLS(simulation.exresult, sm.add_constant(exog)).fit()
        self.metadata = simulation.metadata


    def plot(
        self,
        ax: plt.Axes = None,
        cmap: list = None,
        show_const: bool = False,
        const_color: tuple = plt.cm.gray(.7),
        xscales: np.ndarray = np.array([1.3, 1.8]),
        regex: str = None
    ):
        if regex is not None:
            params = self.result.params.filter(regex=regex)
            pvals = self.result.pvalues.filter(regex=regex)
        elif regex is None and show_const:
            params = self.result.params
            pvals = self.result.pvalues
        else:
            params = self.result.params.drop("const")
            pvals = self.result.pvalues.drop("const")
        
        df = pd.DataFrame(
            params, columns=["coef"]
        ).assign(
            p=pvals
        ).reset_index()

        ax = plt.subplots()[-1] if ax is None else ax

        sns.barplot(
            data=df, y="index", x="coef", 
            palette=[const_color] + list(cmap) if show_const else cmap, 
            ax=ax
        )

        for i, c, p in zip(df.index, df.coef, df.p):
            ax.text(
                sign(c) * (abs(c) + 10), i, asterisk(p) + p_format(p), 
                color=".2", size=7,
                ha="center", va="center"
            )

        ax.set_xlim(xscales * np.array([*ax.get_xlim()]))
        ax.set(
            xlabel="Coefficient", 
            ylabel="", 
            title=f"N={self.metadata['n_rep']}"
        )