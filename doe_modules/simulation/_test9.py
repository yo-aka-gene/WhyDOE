from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._abstract import AbstractSimulator
from doe_modules.design import DOE
from doe_modules.preferences import textcolor
from doe_modules.preferences.cmap import test9


class Test9(AbstractSimulator):
    def __init__(
        self, 
        edge_assignment: list,
        random_state: int = 0,
        model_id: int = "",
        kwarg_v: dict = dict(mean=1, sigma=.8),
        kwarg_a: dict = dict(mean=2, sigma=.3),
        kwarg_b: dict = dict(mean=1, sigma=.5)
    ):
        super().__init__(
            n_factor=9, random_state=random_state, cmap=test9, 
            name="ESM9" if model_id == "" else f"ESM9-#{model_id}"
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
                edge_assignment[:self.n_factor]
            )
        }
        np.random.seed(seeds[2])
        self.b = {
            i: sign * b for i, b, sign in zip(
                [
                    12, 13, 14, 15, 16, 17, 18, 19, 
                    23, 24, 25, 26, 27, 28, 29,
                    34, 35, 36, 37, 38, 39,
                    45, 46, 47, 48, 49,
                    56, 57, 58, 59,
                    67, 68, 69,
                    78, 79, 
                    89
                ],
                np.random.lognormal(**kwarg_b, size=36),
                edge_assignment[self.n_factor:]
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
        
        upstreams = {}
        xs = {}
        idx_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        for i in idx_list:
            upstream_list = [ii for ii in idx_list if ii < i]
            upstream = 0 if len(upstream_list) == 0 else np.sum([self.b[10 * ii + i] * xs[ii] for ii in xs])
            xs = {**xs, i: f(c[i] * (self.v[i] + e[i] + upstream))}
            # print(xs)
        self.c = c
        self.x = xs
        self.y = f(sum([self.a[i + 1] * self.x[i + 1] for i in range(self.n_factor)]) + e[0])
        return self.y
    

    def plot(
        self,
        ax: plt.Axes
    ):
        def curve(p1, p2, h, cap: bool):
            A, B = p1
            C, D = p2
            kb = (B - h)
            kd = (D - h)
            fa = (kb - kd) / kb
            fb = (A * kd - C * kb) / kb
            fc = (kb * C ** 2 - kd * A ** 2) / kb
            rt = np.sqrt(fb ** 2 - fa * fc)
            p = (-fb - rt) / fa if cap else (-fb + rt) / fa
            a = kb / (A - p) ** 2
            return np.vectorize(lambda x: a * (x - p) ** 2 + h)

        def get_top(p1, p2, h, cap: bool):
            A, B = p1
            C, D = p2
            kb = (B - h)
            kd = (D - h)
            fa = (kb - kd) / kb
            fb = (A * kd - C * kb) / kb
            fc = (kb * C ** 2 - kd * A ** 2) / kb
            rt = np.sqrt(fb ** 2 - fa * fc)
            p = (-fb - rt) / fa if cap else (-fb + rt) / fa
            return (p, h)

        dat = pd.DataFrame(dict(x=np.arange(2, 11), y=np.arange(9, 0, -1)))

        sns.scatterplot(
            data=dat,
            x="x",
            y="y", 
            c=test9, s=300, lw=0,
            ax=ax
        )

        fmt_curve = lambda p1, p2, h: (np.linspace(p1[0], p2[0]), [curve(p1, p2, h, True)(v) for v in np.linspace(p1[0], p2[0])])


        box_top = -2
        box_bottom = -3

        ax.fill_between([1.5, 10.5], [box_bottom] * 2, [box_top] * 2, color=".7", alpha=.2)
        ax.text(6, (box_top + box_bottom) / 2, "output value", ha="center", va="center")

        for i in range(9):
            x, y = dat.x[i], dat.y[i]
            alpha = np.linspace(0.5, 1, 9)[i]
            ax.text(x, y, "$x_" + f"{(i + 1)}" + "$", va="center", ha="center", c=textcolor(test9[i]))
            
            if self.a[i + 1] != 0:
                ax.vlines(x, box_top, y, color=test9[i], zorder=-100, alpha=alpha)
                ax.scatter(
                    x, box_top + .25 if self.a[i + 1] > 0 else box_top, 
                    color=test9[i], marker="v" if self.a[i + 1] > 0 else "$-$"
                )
                ax.text(x, box_top + 1, "$a_" + f"{(i + 1)}" + "$", va="center", ha="left")

            for ii, j in enumerate(np.arange(i + 1, 9)):
                if self.b[10 * (i + 1) + j + 1] != 0:
                    x2, y2 = dat.x[j], dat.y[j]
                    if ii == 0:
                        ax.plot([x, x2], [y, y2], c=test9[i], zorder=-100, alpha=alpha)
                        ax.scatter(
                            x2 - .39 if self.b[10 * (i + 1) + j + 1] > 0 else x2 - .33, 
                            y2 + .5 if self.b[10 * (i + 1) + j + 1] > 0 else y2 + .35, 
                            color=test9[i], marker="^" if self.b[10 * (i + 1) + j + 1] > 0 else "$|$",
                            alpha=alpha
                        )
                        ax.text((x + x2) / 2, y2, "$b_{" + f"{(i + 1)}{(j + 1)}" + "}$", va="top", ha="center")
                    else:
                        h = y + ii + 0.2 * i + 1
                        ax.plot(*fmt_curve((x,y), (x2,y2), h), zorder=-100, c=test9[i], alpha=alpha)
                        ax.scatter(
                            x2 - .15 if self.b[10 * (i + 1) + j + 1] > 0 else x2 - .1, 
                            y2 + .86 if self.b[10 * (i + 1) + j + 1] > 0 else y2 + .7, 
                            color=test9[i], marker="<" if self.b[10 * (i + 1) + j + 1] > 0 else "$-$",
                            alpha=alpha
                        )
                        ax.text(*get_top((x,y), (x2,y2), h, True), "$b_{" + f"{(i + 1)}{(j + 1)}" + "}$", va="bottom", ha="center")

        ax.set_xlim(1.5, 10.5)
        ax.set_ylim(-3.0, 18)
        ax.axis("off");
        
#  
#         return None


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
