from itertools import combinations
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import bootstrap

from ._abstract import AbstractSimulator
from doe_modules.design import FullFactorial
from doe_modules.preferences import kwarg_bootstrap as kwarg_bs


class TheoreticalEffects:
    def __init__(
        self,
        simulation: AbstractSimulator,
        interactions: bool = False,
        order: int = np.inf,
        random_state: int = 0,
        n_rep: int = 100,
        model_kwargs: dict = {},
    ):
        assert issubclass(type(simulation), AbstractSimulator), \
            f"pass subclass of AbstractSimulator, got {simulation}[{type(simulation)}]"
        # assert simulation.is_executed, \
        #     f"Simluation is not excecuted yet. Run simulation() before passing it to TheoreticalEffects.__init__"
        np.random.seed(random_state)
        seeds = np.random.randint(0, 2**32, n_rep)
        effects = []
        order = min(order, simulation.n_factor)
        for s in seeds:
            simulation.simulate(
                design=FullFactorial, n_rep=1, random_state=s, 
                model_kwargs=model_kwargs
            )
            mat = simulation.exmatrix.values
            y = simulation.exresult
            effects += [(mat[:, idx] * y).mean() for idx in np.arange(simulation.n_factor)]

            if interactions:
                for n_i in np.arange(2, order + 1):
                    effects += [
                        (self._prod(mat, *idxs) * y).mean() 
                        for idxs in combinations(np.arange(simulation.n_factor), n_i)
                    ]
                
        term_names = eval(
            f"FullFactorial().get_exmatrix(simulation.n_factor){'.interactions(order)' if interactions else '()'}"
        )

        self.result = pd.DataFrame({
            "Coefficient": effects,
            "term": term_names.columns.tolist() * n_rep
        })
        self.cmap = simulation.cmap
        self.metadata = {
            **simulation.metadata,
            "interactions": interactions,
            "order": order,
            "random_state": random_state,
            "n_rep": n_rep,
            "model_kwargs": model_kwargs
        }


    def _prod(self, matrix: np.ndarray, *idxs: Tuple[int]):
        ret = 1
        for i in idxs:
            ret *= matrix[:, i]
        return ret


    def _identifier(self, bootstrap_result, dtype):
        ci = np.array([
            bootstrap_result.confidence_interval[0],
            bootstrap_result.confidence_interval[1]
        ])
        id_idx = np.sum([int((ci > 0)[1]), int((ci > 0).all())]) - 1
        return ["N.S.", "Up", "Down"][id_idx] if dtype == str else dtype(id_idx)


    def summary(
        self,
        dtype: type = str,
        kwarg_bootstrap: dict = kwarg_bs
    ):
        self.metadata = {
            **self.metadata,
            "kwarg_bootstrap": kwarg_bootstrap
        }
        return pd.Series(
            [
                self._identifier(
                    bootstrap(
                        (self.result[self.result.term == factor].Coefficient.values, ),
                        **kwarg_bootstrap
                    ),
                    dtype
                ) for factor in self.result["term"].unique()
            ],
            index=self.result["term"].unique()
        )


    def plot(
        self,
        ax: plt.Axes = None,
        cmap: list = None,
        show_interactions: bool = True,
        order: int = np.inf,
        xscales: np.ndarray = np.array([1.5, 1.05]),
        jitter_ratio: float = .025, 
        regex: str = None,
        size: int = .5,
        show_legend: bool = False,
        **kwargs
    ):
        displayed_factors = self.metadata["factor_list"].copy()
        show_interactions = show_interactions and self.metadata["interactions"]
        order = min(order, self.metadata["order"], len(self.metadata["factor_list"]))
        kwarg_bootstrap = self.metadata["kwarg_bootstrap"] if "kwarg_bootstrap" in self.metadata else kwarg_bs
        
        data = self.result
        directionality = self.summary(
            dtype=int,
            kwarg_bootstrap=kwarg_bootstrap
        )
        coef = data.groupby("term").mean()
        
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
            data = data[data["term"].str.contains(regex)]
            directionality = directionality.loc[displayed_factors]
            coef = coef.loc[displayed_factors]

        ax = plt.subplots()[-1] if ax is None else ax
        cmap = self.cmap if cmap is None else cmap
        
        sns.stripplot(
            data=data, x="Coefficient", y="term", 
            size=size,
            hue="term", ax=ax, palette=cmap
        )
        
        sns.barplot(
            data=data, x="Coefficient", y="term", ax=ax,
            hue="term", legend=show_legend,
            palette=cmap, edgecolor=cmap,
            alpha=0,
            n_boot=self.metadata["kwarg_bootstrap"]["n_resamples"],
            **kwargs
        )
        
        xlims = np.array(ax.get_xlim())
        coef = coef.values.ravel()
        sign = lambda z: int(z / abs(z))

        for i, v in enumerate(directionality.values.ravel()):
            coef_i = coef[i]
            jitter = np.diff(xlims) * jitter_ratio
            jitter *= (2 * np.abs(v) - 1) * sign(coef_i)
            x = coef_i + jitter
            ax.text(
                x, i,
                ["N.S.", "Up", "Down"][v], 
                va="center", 
                ha=["_", "left", "right"][sign(x)],
                size=8
            )

        ax.set_xlim(xlims * np.array(xscales))
        ax.set(ylabel="")
        
        ylim = ax.get_ylim()
        ax.vlines(0, *ylim, linewidth=.5, color=".7", linestyle="--")
        ax.set_ylim(ylim)
        
        # if not show_legend:
        #     ax.legend().remove()
