from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._abstract import AbstractSimulator
from doe_modules.design import DOE
from doe_modules.preferences import textcolor
from doe_modules.preferences.cmap import circuit


class CircuitPlus(AbstractSimulator):
    def __init__(
        self, 
        random_state: int = 0,
        kwarg_v: dict = dict(mean=1, sigma=.8),
        kwarg_a: dict = dict(mean=2, sigma=.3),
        kwarg_b: dict = dict(mean=1, sigma=.5)
    ):
        super().__init__(n_factor=9, random_state=random_state, cmap=circuit, name=r"Model $\Psi^+$")
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
                [1, 2, 7, 8],
                np.random.lognormal(**kwarg_a, size=4)
            )
        }
        np.random.seed(seeds[2])
        self.b = {
            i: b for i, b in zip(
                [34, 35, 36, 37, 45, 46, 47, 56, 57, 67, 78, 89],
                np.random.lognormal(**kwarg_b, size=12)
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
        f = lambda x: max(0, x)
        c = self.c

        x1 = f(c[1] * (self.v[1] + e[1]))
        x2 = f(c[2] * (self.v[2] + e[2]))
        x3 = f(c[3] * (self.v[3] + e[3]))
        x4 = f(c[4] * (self.v[4] + self.b[34] * x3 + e[4]))
        x5 = f(
            c[5] * (self.v[5] - (self.b[35] * x3) + self.b[45] * x4 + e[5])
        )
        x6 = f(
            c[6] * (self.v[6] + self.b[36] * x3  - (self.b[46] * x4) + self.b[56] * x5 +  e[6])
        )
        x7 = f(
            c[7] * (self.v[7] + self.b[37] * x3 + self.b[47] * x4 + self.b[57] * x5 + self.b[67] * x6 + e[7])
        )
        x8 = f(c[8] * (self.v[8] + self.b[78] * x7 + e[8]))
        x9 = f(c[9] * (self.v[9] + self.b[89] * x8 + e[9]))
        self.c = c
        self.x = {i+1: xi for i, xi in enumerate([x1, x2, x3, x4, x5, x6, x7, x8, x9])}
        self.y = f(self.a[1] * x1 - (self.a[2] * x2) + self.a[7] * x7 + self.a[8] * x8 + e[0])
        return self.y
    

    def plot(
        self,
        ax: plt.Axes
    ):
        ax.fill_between([0, 10], [0, 0], [1, 1], color=".7", alpha=.2)

        datdot = pd.DataFrame({
            "name": [
                f"X{i + 1}" if self.x is None else round(self.x[i + 1], 2) for i in range(9)
            ],
            "h": [1, 3, 3, 7, 1, 9, 5, 7, 9],
            "v": [3, 3, 9, 9, 6, 6, 3, 3, 3],
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
        ax.annotate("", [3, 1.1], [3, 3], arrowprops=ap2(2), **arrconf)
        ax.annotate("", [6.5, 9], [3, 9], arrowprops=ap(3), **arrconf)
        ax.annotate("", [1.4, 6.6], [3, 9], arrowprops=ap2(3), **arrconf)
        ax.annotate("", [8.4, 6.3], [3, 9], arrowprops=ap(3), **arrconf)
        ax.annotate("", [4.7, 3.9], [3, 9], arrowprops=ap(3), **arrconf)
        ax.annotate("", [1.6, 6.3], [7, 9], arrowprops=ap(4), **arrconf)
        ax.annotate("", [8.6, 6.6], [7, 9], arrowprops=ap2(4), **arrconf)
        ax.annotate("", [5.3, 3.9], [7, 9], arrowprops=ap(4), **arrconf)
        ax.annotate("", [8.5, 6], [1, 6], arrowprops=ap(5), **arrconf)
        ax.annotate("", [4.4, 3.3], [1, 6], arrowprops=ap(5), **arrconf)
        ax.annotate("", [5.6, 3.3], [9, 6], arrowprops=ap(6), **arrconf)
        ax.annotate("", [5, 1], [5, 3], arrowprops=ap(7), **arrconf)
        ax.annotate("", [6.5, 3], [5, 3], arrowprops=ap(7), **arrconf)
        ax.annotate("", [7, 1], [7, 3], arrowprops=ap(8), **arrconf)
        ax.annotate("", [8.5, 3], [7, 3], arrowprops=ap(8), **arrconf)
        

        for i in range(len(datdot)):
            ax.text(
                *datdot.iloc[i, 1:3], datdot.iloc[i, 0], ha="center", va="center",
                size="medium" if self.x is None else 7, 
                c=textcolor(datdot.c[i])
            )

        for n, a in self.a.items():
            ax.text(
                datdot.iloc[n - 1, 1], 1.75, "" if datdot.name[n - 1] in [f"X{n}", 0.0] else r"$\times$" + f"{a.round(2)}",
                ha="center", va="center", size=6
            )
        
        for n, b in self.b.items():
            ax.text(
                {34: 5, 35: 2, 36: 4, 37: 3.5, 45: 6, 46: 8, 47: 6.5, 56: 5, 57: 3, 67: 7, 78: 6, 89: 8}[n],
                {34: 9, 35: 7.5, 36: 8.5, 37: 8, 45: 8.5, 46: 7.5, 47: 8, 56: 6, 57: 4.5, 67: 4.5, 78: 3, 89: 3}[n],
                "" if datdot.name[int(str(n)[0]) - 1] in [f"X{int(str(n)[0])}", 0.0] else r"$\times$" + f"{b.round(2)}",
                ha="center", va="center", size=6
            )
        
        if self.c is not None:
            for n, c in self.c.items():
                ax.scatter(
                    *{i + 1: (h, v) for i, h, v in zip(range(9), datdot.h, datdot.v)}[n],
                    marker="x", s=300, color="r"
                ) if c == 0 else None

        ax.text(5, .5, "output value" if self.y is None else round(self.y, 2), ha="center", va="center")

        ax.set_ylim([0, 10])
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