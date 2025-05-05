import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

from ._abstract import AbstractSimulator
from doe_modules.design import DesignMatrix
from doe_modules.preferences.pvalues import sign, asterisk, p_format


class MLR:
    def __init__(
        self,
        simulation: AbstractSimulator,
        interactions: bool = False,
        order: int = 2,
        full_model: bool = False,
    ):
        assert issubclass(type(simulation), AbstractSimulator), \
            f"pass subclass of AbstractSimulator, got {simulation}[{type(simulation)}]"
        assert simulation.is_executed, \
            f"Simluation is not excecuted yet. Run simulation.simulate before passing it to MLR.__init__"
        exog = DesignMatrix(simulation.exmatrix.values).interactions(order=order, full=full_model) if interactions else simulation.exmatrix
        self.result = ols(
            "y ~ " + " + ".join(exog.columns), 
            exog.assign(y=simulation.exresult)
        ).fit()
        self.cmap = simulation.cmap
        self.metadata = {
            **simulation.metadata,
            "interactions": interactions,
            "order": order,
            "full_model": full_model
        }


    def plot(
        self,
        ax: plt.Axes = None,
        cmap: list = None,
        show_const: bool = False,
        const_color: tuple = plt.cm.gray(.7),
        xscales: np.ndarray = np.array([1.3, 1.8]),
        anova: bool = False,
        anova_type: int = 2,
        jitter_ratio: float = .1,
        regex: str = None
    ):
        params, pvals = self._overwrite_with_anova(
            show_const=show_const,
            anova=anova,
            anova_type=anova_type,
            regex=regex
        )
        
        df = pd.DataFrame(
            params, columns=["coef"]
        ).assign(
            p=pvals
        ).reset_index()

        ax = plt.subplots()[-1] if ax is None else ax
        cmap = self.cmap if cmap is None else cmap

        sns.barplot(
            data=df, y="index", x="coef", 
            palette=[const_color] + list(cmap) if show_const else cmap, 
            ax=ax
        )
        
        xm, xM = ax.get_xlim()
        jitter = jitter_ratio * (xM - xm)

        for i, c, p in zip(df.index, df.coef, df.p):
            ax.text(
                sign(c) * (abs(c) + jitter), i, asterisk(p) + p_format(p), 
                color=".2", size=7,
                ha="center", va="center"
            )

        ax.set_xlim(xscales * np.array([*ax.get_xlim()]))
        ax.set(
            xlabel="Coefficient", 
            ylabel="", 
            title=f"N={self.metadata['n_rep']}"
        )


    def summary(
        self,
        dtype: type = str,
        alpha: float = .05,
        show_const: bool = False,
        anova: bool = False,
        anova_type: int = 2,
        regex: str = None
    ) -> pd.Series:
        assert dtype in [str, int, float], \
            f"Invalid dtype while only str, int, or float are supported; got {dtype}"
        params, pvals = self._overwrite_with_anova(
            show_const=show_const,
            anova=anova,
            anova_type=anova_type,
            regex=regex
        )
        ret = params.apply(
            lambda x: 1 if x >= 0 else -1
        ) * pvals.apply(
            lambda p: 0 if p >= alpha else 1
        ) * pvals.apply(
            lambda p: 2 if np.isnan(p) else 1
        )
        return ret.apply(
            lambda i: ["N.S.", "Up", "N/A", "Down"][i]
        ) if dtype == str else ret.apply(
            lambda x: np.nan if np.abs(x) == 2 else dtype(x)
        )

    
    def _overwrite_with_anova(
        self,
        show_const: bool = False,
        anova: bool = False,
        anova_type: int = 2,
        regex: str = None
    ) -> (pd.Series, pd.Series):
        if anova:
            params = self.result.params.drop("Intercept")
            try:
                pvals = sm.stats.anova_lm(
                    self.result, typ=anova_type
                ).loc[:, "PR(>F)"].drop("Residual")
            except ValueError:
                pvals = pd.Series(
                    [np.nan] * params.size,
                    index = params.index
                )
        else:
            params = self.result.params
            pvals = self.result.pvalues

        if regex is not None:
            params = params.filter(regex=regex)
            pvals = pvals.filter(regex=regex)
        elif regex is None and not show_const and not anova:
            params = params.drop("Intercept")
            pvals = pvals.drop("Intercept")
        else:
            pass
        return params, pvals
