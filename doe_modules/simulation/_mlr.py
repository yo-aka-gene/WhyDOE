from itertools import combinations

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
    ):
        assert issubclass(type(simulation), AbstractSimulator), \
            f"pass subclass of AbstractSimulator, got {simulation}[{type(simulation)}]"
        assert simulation.is_executed, \
            f"Simluation is not excecuted yet. Run simulation.simulate before passing it to MLR.__init__"
        exog = DesignMatrix(simulation.exmatrix.values).interactions(order=order) if interactions else simulation.exmatrix
        self.result = ols(
            "y ~ " + " + ".join(exog.columns), 
            exog.assign(y=simulation.exresult)
        ).fit()
        self.cmap = simulation.cmap
        self.metadata = {
            **simulation.metadata,
            "interactions": interactions,
            "order": order
        }


    def plot(
        self,
        ax: plt.Axes = None,
        cmap: list = None,
        show_const: bool = False,
        show_interactions: bool = True,
        order: int = np.inf,
        const_color: tuple = plt.cm.gray(.7),
        xscales: np.ndarray = np.array([1.1, 1.1]),
        anova: bool = False,
        anova_type: int = 2,
        jitter_ratio: float = .1,
        regex: str = None,
        display_pvals: bool = False
    ):
        params, pvals = self._overwrite_with_anova(
            show_const=show_const,
            anova=anova,
            anova_type=anova_type,
            regex=regex
        )
        
        displayed_factors = ["Intercept"] + self.metadata["factor_list"] if show_const else self.metadata["factor_list"].copy()
        show_interactions = show_interactions and self.metadata["interactions"]
        order = min(order, self.metadata["order"], len(self.metadata["factor_list"]))
        
        if show_interactions and order >= 2:
            for i in np.arange(2, order + 1):
                displayed_factors += list(
                    map(
                        lambda tup: "".join(tup), 
                        combinations(self.metadata["factor_list"], i)
                    )
                )
        if regex is not None:
            find_regex = lambda arr, key: arr[np.vectorize(lambda element: key in element)(arr)]
            displayed_factors = find_regex(np.array(displayed_factors), regex)
        
        df = pd.DataFrame(
            params, columns=["coef"]
        ).assign(
            p=pvals
        ).assign(
            neglog10=-np.log2(pvals.fillna(np.finfo(float).eps))
        ).loc[displayed_factors, :].reset_index()

        ax = plt.subplots()[-1] if ax is None else ax
        cmap = self.cmap if cmap is None else cmap
        
        sns.scatterplot(
            data=df, y="index", x="coef",
            palette=[const_color] + list(cmap) if show_const else cmap, 
            ax=ax, hue="index", legend=False, size="neglog10"
        )
        
        for (i, x), c in zip(enumerate(params), cmap):
            ax.hlines(i, 0, x, color=c)
        
        ax.set_xlim([-1, 1])
        ylim = ax.get_ylim()
        ax.vlines(0, *ylim, color=".2", linewidth=.5, zorder=-300)
        ax.set_ylim(*ylim)
        
        xm, xM = ax.get_xlim()
        jitter = jitter_ratio * (xM - xm)

        for i, c, p in zip(df.index, df.coef, df.p):
            ax.text(
                sign(c) * (abs(c) + jitter), i, 
                asterisk(p) + p_format(p) if display_pvals else asterisk(p), 
                color=".2", size=7,
                ha="center", va="center"
            )

        ax.set_xlim(xscales * np.array([*ax.get_xlim()]))
        ax.set(
            xlabel="Coefficient" if not anova else r"sign$(\hat{\beta})\eta_p^2$", 
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
        fill_nan: bool = False,
        regex: str = None
    ) -> pd.Series:
        assert dtype in [str, int, float], \
            f"Invalid dtype while only str, int, or float are supported; got {dtype}"
        params, pvals = self._overwrite_with_anova(
            show_const=show_const,
            anova=anova,
            anova_type=anova_type,
            fill_nan=fill_nan,
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
        fill_nan: bool = False,
        regex: str = None,
    ) -> (pd.Series, pd.Series):
        if anova:
            coef = self.result.params.drop("Intercept")
            try:
                anova_res = sm.stats.anova_lm(
                    self.result, typ=anova_type
                )
                ss_e = anova_res.loc["Residual", "sum_sq"]
                pvals = anova_res.loc[:, "PR(>F)"].drop("Residual")
                
                eta_sq = lambda df_anova: df_anova.sum_sq / (df_anova.sum_sq + ss_e)
                
                params = anova_res.drop("Residual").assign(
                    coef=eta_sq(anova_res.drop("Residual")) * np.sign(coef)
                )["coef"]

            except ValueError:
                pvals = pd.Series(
                    [np.nan] * coef.size,
                    index = coef.index
                )
                params = pd.Series(
                    [0] * coef.size,
                    index = coef.index
                )
        else:
            params = self.result.params
            pvals = self.result.pvalues
        
        if fill_nan:
            pvals = pvals.fillna(1)

        if regex is not None:
            params = params.filter(regex=regex)
            pvals = pvals.filter(regex=regex)
        elif regex is None and not show_const and not anova:
            params = params.drop("Intercept")
            pvals = pvals.drop("Intercept")
        else:
            pass
        return params, pvals
