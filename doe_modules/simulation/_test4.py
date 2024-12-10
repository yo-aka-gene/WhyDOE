from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._abstract import AbstractSimulator
from doe_modules.design import DOE
from doe_modules.preferences import textcolor
from doe_modules.preferences.cmap import test4


class Test4(AbstractSimulator):
    def __init__(
        self, 
        edge_assignsment: list,
        random_state: int = 0,
        model_id: int = "",
        kwarg_v: dict = dict(mean=1, sigma=.8),
        kwarg_a: dict = dict(mean=2, sigma=.3),
        kwarg_b: dict = dict(mean=1, sigma=.5)
    ):
        super().__init__(
            n_factor=4, random_state=random_state, cmap=test4, 
            name="ESM4" if model_id == "" else f"ESM4-#{model_id}"
        )
        seeds = np.random.randint(0, 2**32, 3)
        np.random.seed(seeds[0])
        self.v = {
            i+1: vi for i, vi in enumerate(
                np.random.lognormal(**kwarg_v, size=self.n_factor)
            )
        }
        np.random.seed(seeds[1])
        self.a = {
            i: sign * a for i, a, sign in zip(
                np.arange(1, self.n_factor + 1).astype(int),
                np.random.lognormal(**kwarg_a, size=self.n_factor),
                edge_assignsment[:self.n_factor]
            )
        }
        np.random.seed(seeds[2])
        self.b = {
            i: sign * b for i, b, sign in zip(
                [12, 13, 14, 23, 24, 34],
                np.random.lognormal(**kwarg_b, size=6),
                edge_assignsment[self.n_factor:]
            )
        }

    def run(
        self, 
        design_array, 
        random_state: int = 0,
        kwarg_err: dict = dict(loc=0, scale=1),
    ):
        super().run(design_array=design_array, random_state=random_state)
        e = np.random.normal(**kwarg_err, size=6)
        f = lambda x: max(0, x)
        c = self.c

        x1 = f(c[1] * (self.v[1] + e[1]))
        x2 = f(c[2] * (self.v[2] + self.b[12] * x1 + e[2]))
        x3 = f(c[3] * (self.v[3] + self.b[13] * x1 + self.b[23] * x2 + e[3]))
        x4 = f(c[4] * (self.v[4] + self.b[14] * x1 + self.b[24] * x2 + self.b[34] * x3 + e[4]))
        self.c = c
        self.x = {i+1: xi for i, xi in enumerate([x1, x2, x3, x4])}
        self.y = sum([self.a[i + 1] * self.x[i + 1] for i in range(self.n_factor)]) + e[0]
        return self.y
    

    def plot(
        self,
        ax: plt.Axes
    ):
        ax.fill_between([0, 10], [0, 0], [1, 1], color=".7", alpha=.2)

        datdot = pd.DataFrame({
            "name": [
                f"X{i + 1}" if self.x is None else round(self.x[i + 1], 2) for i in range(self.n_factor)
            ],
            "h": [2, 4, 6, 8],
            "v": [9, 7, 5, 3],
            "c": [
               ".7" if self.x is not None and self.x[i + 1] == 0 else self.cmap[i] for i in range(self.n_factor)
            ],
            "alpha": [
               .2 if self.x is not None and self.x[i + 1] == 0 else 1 for i in range(self.n_factor)
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
            arrowstyle="|-|", connectionstyle="arc3",
            facecolor=datdot.c[n - 1], edgecolor=datdot.c[n - 1],
            linewidth=2, mutation_scale=4,
            alpha=datdot.alpha[n - 1]
        )
        
        angleA = lambda n: [0, -90, 0, -45][n - 1]
        angleB = lambda n: [90, 180, 105, 120][n - 1]
        
        app = lambda n: dict(
            shrink=0, width=1, headwidth=5, 
            headlength=5, 
            connectionstyle=f"angle3, angleA={angleA(n)}, angleB={angleB(n)}",
            facecolor=datdot.c[n - 1], edgecolor=datdot.c[n - 1],
            alpha=datdot.alpha[n - 1]
        )
        
        apn = lambda n: dict(
            arrowstyle="|-|", connectionstyle=f"angle3, angleA={angleA(n)}, angleB={angleB(n)}",
            facecolor=datdot.c[n - 1], edgecolor=datdot.c[n - 1],
            linewidth=2, mutation_scale=4,
            alpha=datdot.alpha[n - 1]
        )

        arrconf=dict(ha="center", va="center", zorder=-10)
        
        edge_pos = {
            12: ([datdot.h[1] - .5 / np.sqrt(2), datdot.v[1] + .5 / np.sqrt(2)], [datdot.h[0], datdot.v[0]]),
            13: ([datdot.h[2], datdot.v[2] + .5], [datdot.h[0], datdot.v[0]]),
            14: ([datdot.h[3], datdot.v[3] + .75], [datdot.h[0], datdot.v[0]]),
            23: ([datdot.h[2] - .5 / np.sqrt(2), datdot.v[2] + .5 / np.sqrt(2)], [datdot.h[1], datdot.v[1]]),
            24: ([datdot.h[3] - .5, datdot.v[3]], [datdot.h[1], datdot.v[1]]),
            34: ([datdot.h[3] - .5 / np.sqrt(2), datdot.v[3] + .5 / np.sqrt(2)], [datdot.h[2], datdot.v[2]]),
        }
        b_pos = {
            12: (datdot.h[0:2].mean(), datdot.v[0:2].mean()),
            13: (datdot.h[2], datdot.v[1]),
            14: (datdot.h[3], datdot.v[2]),
            23: (datdot.h[1:3].mean(), datdot.v[1:3].mean()),
            24: (datdot.h[1:3].mean(), datdot.v[2:4].mean()),
            34: (datdot.h[2:4].mean(), datdot.v[2:4].mean()),
        }
        
        a_aps = {
            k: ap(k) if a >=0 else ap2(k) for k, a in self.a.items()
        }

        b_aps = {
            12: ap(1) if self.b[12] >=0 else ap2(1),
            13: app(1) if self.b[13] >=0 else apn(1),
            14: app(1) if self.b[14] >=0 else apn(1),
            23: ap(2) if self.b[23] >=0 else ap2(2),
            24: app(2) if self.b[24] >=0 else apn(2),
            34: ap(3) if self.b[34] >=0 else ap2(3),
        }
        
        for idx, a in self.a.items():
            if a != 0:
                ax.annotate("", [datdot.h[idx - 1], 1], [datdot.h[idx - 1], datdot.v[idx - 1]], arrowprops=a_aps[idx], **arrconf)
                ax.text(
                    datdot.h[idx - 1], 2,
                    "" if datdot.name[int(str(idx)[0]) - 1] in [f"X{int(str(idx)[0])}", 0.0] else r"$\times$" + f"{a.round(2)}",
                    ha="center", va="center", size=6
                )
        
        for idx, b in self.b.items():
            if b != 0:
                ax.annotate("", *edge_pos[idx], arrowprops=b_aps[idx], **arrconf)
                ax.text(
                    *b_pos[idx],
                    "" if datdot.name[int(str(idx)[0]) - 1] in [f"X{int(str(idx)[0])}", 0.0] else r"$\times$" + f"{b.round(2)}",
                    ha="center", va="center", size=6
                )

        for i in range(len(datdot)):
            ax.text(
                *datdot.iloc[i, 1:3], datdot.iloc[i, 0], ha="center", va="center",
                size="medium" if self.x is None else 7,
                c=textcolor(datdot.c[i])
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