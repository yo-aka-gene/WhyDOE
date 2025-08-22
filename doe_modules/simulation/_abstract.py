from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from doe_modules.design import DOE
from doe_modules.preferences import subplots


class AbstractSimulator:
    def __init__(
        self, 
        n_factor: int, 
        random_state: int, 
        cmap: list,
        name: str
    ):
        self.n_factor = n_factor
        self.is_executed = False
        self.design = None
        self.metadata = {}
        self.cmap = cmap
        self.name = name
        self.c, self.x, self.y = None, None, None
        np.random.seed(random_state)

    def run(
        self, 
        design_array: np.ndarray, 
        random_state: int = 0
    ):
        self.c = {
            i+1: (ci + 1) / 2 for i, ci in enumerate(design_array)
        }
        np.random.seed(random_state)
    

    def plot(
        self,
        ax: plt.Axes
    ):
        pass


    def simulate(
        self,
        design: DOE = None,
        n_rep: int = 1,
        random_state: int = 0,
        plot: bool = False,
        ax: np.ndarray = None,
        titles: List[str] = None,
        model_kwargs: dict = None,
        **kwargs
    ):
        self.design = self.design if self.is_executed and design is None else design
        assert issubclass(type(self.design()), DOE), \
            f"Assign valid design, got {design}[{type(design)}]"
        self.design = self.design()
        self.is_executed = True
        self.metadata = {
            "design": self.design.name, 
            "n_rep": n_rep,
            "factor_list": self.design.get_exmatrix(self.n_factor, **kwargs)().columns.tolist(),
            "kwargs": kwargs
        }
        self.exmatrix = pd.concat([self.design.get_exmatrix(self.n_factor, **kwargs)()] * n_rep)
        np.random.seed(random_state)
        seeds = np.random.randint(0, 2**32, len(self.exmatrix))
        exresult = []
        titles = [
            f"#{i + 1}" for i in range(len(self.exmatrix))
        ] if titles is None else titles

        if plot:
            ax = subplots(len(seeds))[1] if ax is None else ax
        
        for i, s in enumerate(seeds):
            exresult += [
                self.run(
                    design_array=self.exmatrix.iloc[i, :], 
                    random_state=s,
                    **model_kwargs
                )
            ]

            if plot:
                self.plot(ax=ax.ravel()[i])
                ax.ravel()[i].set(title=titles[i]) 

        self.exresult = np.array(exresult)

        if plot:
            [a.axis("off") for a in ax.ravel()[len(self.exmatrix) - ax.ravel().size:]];


    def scatterview(
        self, 
        ax: np.ndarray = None,
    ):
        assert self.is_executed, \
            f"Simluation is not excecuted yet. Run self.simulate before calling self.scatterview"
        ax = subplots(self.n_factor)[1] if ax is None else ax
        factors = [f"X{i + 1}" for i in range(self.n_factor)]
        n_trials = len(self.design.get_exmatrix(self.n_factor, **self.metadata['kwargs'])())
        
        for i, a in enumerate(ax.ravel()):
            factor = factors[i]
            df = pd.DataFrame({
                factor: [
                    ["$-$", "$+$"][v] for v in ((1 + self.exmatrix.loc[:, factor]) / 2).astype(int)
                ],
                "idx": self.exmatrix.reset_index().loc[:, "index"]
            })
            sns.scatterplot(
                data=df, y="idx", x=self.exresult, s=30, hue=factor, 
                palette={"$+$": "C3", "$-$":"C0"}, ax=a
            )
            
            get = lambda val: self.exresult[np.where(self.exmatrix.loc[:, factor] == val)]
            summary = lambda arr: (np.mean(arr), np.max(arr), np.min(arr))

            mean_pos, max_pos, min_pos = summary(get(1))
            mean_neg, max_neg, min_neg = summary(get(-1))

            a.scatter(
                x=[mean_pos, mean_neg],
                y=[n_trials, n_trials + 1], marker=",", color=["C3", "C0"], s=50
            )

            a.hlines(n_trials, min_pos, max_pos, color="C3")
            a.hlines(n_trials + 1, min_neg, max_neg, color="C0")
            a.set(
                xlabel="output values", ylabel="", 
                title=factor
            )
            handles, labels = a.get_legend_handles_labels()
            a.legend(
                handles = handles if df.loc[:, factor][0] == "$+$" else handles[::-1],
                labels = labels if df.loc[:, factor][0] == "$+$" else labels[::-1],
                loc="center right", bbox_to_anchor=(1, .5)
            )
        