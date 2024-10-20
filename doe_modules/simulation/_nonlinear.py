from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import gelu

from ._abstract import AbstractSimulator
from doe_modules.design import DOE
from doe_modules.preferences.cmap import nonlinear


class NonLinear(AbstractSimulator):
    def __init__(
        self, 
        random_state: int = 0,
        kwarg_v: dict = dict(mean=1, sigma=.8),
        kwarg_a: dict = dict(mean=2, sigma=.3),
        kwarg_b: dict = dict(mean=1, sigma=.5)
    ):
        super().__init__(n_factor=9, random_state=random_state, cmap=nonlinear, name=r"Model $\Phi^*$")
        seeds = np.random.randint(0, 2**32, 3)
        np.random.seed(seeds[0])
        self.v = {
            i+1: vi for i, vi in enumerate(
                np.random.lognormal(**kwarg_v, size=9)
            )
        }
        np.random.seed(seeds[1])
        self.a = {
            i: a for i, a in zip(
                [1, 2, 6, 7, 9],
                np.random.lognormal(**kwarg_a, size=5)
            )
        }
        np.random.seed(seeds[2])
        self.b = {
            i: b for i, b in zip(
                [2, 3, 4, 5, 6, 8],
                np.random.lognormal(**kwarg_b, size=6)
            )
        }

    def run(
        self, 
        design_array, 
        random_state: int = 0,
        kwarg_err: dict = dict(loc=0, scale=1),
    ):
        super().run(design_array=design_array, random_state=random_state)
        e = np.random.normal(**kwarg_err, size=10)
        f = lambda x: x * (np.tanh(x) + 1) / 2
        g = lambda x: gelu(torch.tensor(x)).item()
        logistic_like = lambda x: 1 / (1 + np.e ** (-x / 100))
        c = self.c

        x1 = f(c[1] * (self.v[1] + e[1]))
        x2 = f(c[2] * (self.v[2] + e[2]))
        x3 = f(c[3] * (self.v[3] + e[3]))
        x4 = f(c[4] * (self.v[4] + e[4]))
        x5 = f(
            c[5] * (self.v[5] - self.b[4] * x4 + e[5])
        )
        x6 = f(c[6] * (self.v[6] + e[6]))
        x7 = f(
            c[7] * (
                self.v[7] + self.b[2] * x2 + self.b[3] * x3 - (self.b[5] * x5) - (self.b[6] * x6) + e[7]
            )
        )
        x8 = f(c[8] * (self.v[8] + e[8]))
        x9 = f(
            c[9] * (self.v[9] + self.b[8] * x8 + e[9])
        )
        self.c = c
        self.x = {i+1: xi for i, xi in enumerate([x1, x2, x3, x4, x5, x6, x7, x8, x9])}
        
        self.y = logistic_like(
            self.a[1] * x1 + self.a[2] * x2 + g(self.a[6] * x6) + g(self.a[7] * x7) - (self.a[9] * x9) + e[0]
        )
        return self.y
    

    def plot(
        self,
        ax: plt.Axes
    ):
        ax.fill_between([0, 10], [0, 0], [1, 1], color=".7", alpha=.2)
        
        ax.fill_between([0, 10], [-2, -2], [-1, -1], color=".7", alpha=.2)

        datdot = pd.DataFrame({
            "name": [
                f"X{i + 1}" if self.x is None else round(self.x[i + 1], 2) for i in range(9)
            ],
            "h": [1, 3, 5, 7, 7, 7, 5, 9, 9],
            "v": [3, 6, 6, 9, 6, 3, 3, 6, 3],
            "c": [
               ".7" if self.x is not None and self.x[i + 1] == 0 else self.cmap[i] for i in range(9)
            ],
            "alpha": [
               .2 if self.x is not None and self.x[i + 1] == 0 else 1 for i in range(9)
            ],
        })

        ax.scatter(data=datdot, x="h", y="v", s=500, color="c", alpha=datdot.alpha)
        ap = lambda n: dict(
            shrink=0, width=1, headwidth=5, 
            headlength=5, connectionstyle="arc3",
            facecolor=datdot.c[n - 1], edgecolor=datdot.c[n - 1],
            alpha=datdot.alpha[n - 1]
        )
        ap2 = lambda n: dict(
            arrowstyle="|-|", 
            facecolor=datdot.c[n - 1], edgecolor=datdot.c[n - 1],
            linewidth=2, mutation_scale=4,
            alpha=datdot.alpha[n - 1]
        )
        arrconf=dict(ha="center", va="center", zorder=-10)

        ax.annotate("", [1, 1], [1, 3], arrowprops=ap(1), **arrconf)
        ax.annotate("", [3, 1], [3, 6], arrowprops=ap(2), **arrconf)
        ax.annotate("", [4.5, 3.5], [3, 6], arrowprops=ap(2), **arrconf)
        ax.annotate("", [5, 3.7], [5, 6], arrowprops=ap(3), **arrconf)
        ax.annotate("", [7, 6.8], [7, 9], arrowprops=ap2(4), **arrconf)
        ax.annotate("", [5.5, 3.7], [7, 6], arrowprops=ap2(5), **arrconf)
        ax.annotate("", [5.6, 3], [7, 3], arrowprops=ap2(6), **arrconf)
        ax.annotate("", [7, 1], [7, 3], arrowprops=ap(6), **arrconf)
        ax.annotate("", [5, 1], [5, 3], arrowprops=ap(7), **arrconf)
        ax.annotate("", [9, 3.7], [9, 6], arrowprops=ap(8), **arrconf)
        ax.annotate("", [9, 1.1], [9, 3], arrowprops=ap2(9), **arrconf)
        
        ax.annotate(
            "", [5, -1], [5, 0], 
            arrowprops=dict(
                shrink=0, width=1, headwidth=5, 
                headlength=5, connectionstyle="arc3",
                facecolor=".7", edgecolor=".7",
                alpha=1
            ), **arrconf
        )

        for i in range(len(datdot)):
            ax.text(
                *datdot.iloc[i, 1:3], datdot.iloc[i, 0], ha="center", va="center",
                size="medium" if self.x is None else 7
            )

        for n, a in self.a.items():
            ax.text(
                datdot.iloc[n - 1, 1], 1.75, "" if datdot.name[n - 1] in [f"X{n}", 0.0] else r"$\times$" + f"{a.round(2)}",
                ha="center", va="center", size=6
            )
        
        for n, b in self.b.items():
            ax.text(
                {2: 4, 3: 5, 4: 7, 5: 6, 6: 6, 8: 9}[n],
                {2: 4.5, 3: 4.5, 4: 7.5, 5: 4.5, 6: 3, 8: 4.5}[n],
                "" if datdot.name[n - 1] in [f"X{n}", 0.0] else r"$\times$" + f"{b.round(2)}",
                ha="center", va="center", size=6
            )
        
        if self.c is not None:
            for n, c in self.c.items():
                ax.scatter(
                    *{i + 1: (h, v) for i, h, v in zip(range(9), datdot.h, datdot.v)}[n],
                    marker="x", s=300, color="r"
                ) if c == 0 else None

        ax.text(5, .5, "non-linear transformation", ha="center", va="center")
        ax.text(5, -1.5, "output value" if self.y is None else round(self.y, 4), ha="center", va="center")

        ax.set_ylim([-2, 10])
        ax.set_xlim([0, 10])

        ax.axis("off")
        return None


    def simulate(
        self,
        design: DOE = None,
        n_rep: int = 1,
        random_state: int = 0,
        plot: bool = False,
        ax: np.ndarray = None,
        titles: List[str] = None,
        model_kwargs: dict = {},
        **kwargs
    ):
        super().simulate(
            design=design, n_rep=n_rep,
            random_state=random_state,
            plot=plot, ax=ax, 
            titles=titles,
            model_kwargs=model_kwargs,
            **kwargs
        )


    def scatterview(
        self,
        ax: plt.Axes
    ):
        super().scatterview(ax=ax)
