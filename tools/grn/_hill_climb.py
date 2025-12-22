import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pgmpy.estimators import HillClimbSearch as PGMPYHC
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from tools.bioinfo import GeneList
from tools.preferences import rgba2gray, textcolor


class HillClimbSearch:
    def __init__(
        self,
        gene_list: GeneList,
        output_value: str,
        metacell: str = "SEACells",
        scoring_method="bic-g"
    ):
        self.data = pd.concat(
            [
                pd.DataFrame(
                    StandardScaler().fit_transform(
                        gene_list.data[:, gene_list.selected_ids].X
                    ),
                    index=gene_list.data.obs_names, 
                    columns=gene_list.selected_genes
                ),
                gene_list.data.obs.loc[:, [output_value, metacell]].rename(
                    columns={output_value: "output", metacell: "MetaCell"}
                )
            ],
            axis=1
        ).groupby("MetaCell", observed=False).mean()

        self.model = PGMPYHC(data=self.data, use_cache=False)
        self.result = self.model.estimate(
            scoring_method=scoring_method
        )
        self.graph = nx.DiGraph(self.result.edges())
        self.subgraph = self.graph.subgraph(
            nx.ancestors(self.graph, "output") | {"output"}
        )
        order = sorted(
            list(self.subgraph.nodes()),
            key=lambda n: len(nx.ancestors(self.subgraph, n))
        )
        self.adjacent_matrix = nx.to_pandas_adjacency(
            self.subgraph, dtype=int
        ).loc[order, order].drop("output")
        self.grn_arr = create_grn_arr(
            self.adjacent_matrix * np.sign(
                self.data.loc[:, self.adjacent_matrix.columns].corr().drop("output")
            )
        )


    def draw_graph(
        self,
        output_label: str = "output",
        output_color: str = "lightpink",
        node_color: str = "lavender",
        **kwargs
    ):
        conversion = lambda k: k if k != "output" else output_label
        G = nx.DiGraph([
            (conversion(u), conversion(v)) for u, v in self.result.edges()
        ])
        pos = nx.circular_layout(G)
        node_color = [
            output_color if k == output_label else node_color for k in pos
        ]
        nx.draw(G, pos, node_color=node_color, **kwargs)


    def draw_grn(
        self,
        ax: plt.Axes,
        output_label: str = "output value",
        markersize: int = 1000,
        fontsize: int = "small",
        pathway_loc: float = .2,
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

        n_factor = len(self.subgraph.nodes()) - 1

        dat = pd.DataFrame(
            dict(x=np.arange(2, 2 + n_factor), y=np.arange(n_factor, 0, -1))
        )

        cmap = [plt.cm.jet((i + 2)/(n_factor + 2)) for i in range(n_factor)]

        sns.scatterplot(
            data=dat,
            x="x",
            y="y", 
            c=cmap, 
            s=markersize, lw=0,
            ax=ax
        )

        fmt_curve = lambda p1, p2, h: (
            np.linspace(p1[0], p2[0]), 
            [curve(p1, p2, h, True)(v) for v in np.linspace(p1[0], p2[0])]
        )

        box_top = -2
        box_bottom = -3

        ax.fill_between(
            [1.5, 1.5 + n_factor], [box_bottom] * 2, [box_top] * 2, color=".7", alpha=.2
        )
        ax.text(
            1.5 + (n_factor / 2), (box_top + box_bottom) / 2, 
            output_label, ha="center", va="center"
        )

        height = 0

        for i in range(n_factor):
            x, y = dat.x[i], dat.y[i]
            alpha = np.linspace(0.5, 1, n_factor)[i]
            ax.text(
                x, y, self.adjacent_matrix.index[i], 
                va="center", ha="center", c=textcolor(cmap[i]),
                fontsize=fontsize
            )
            
            if self.grn_arr[i] != 0:
                ax.vlines(x, box_top, y, color=cmap[i], zorder=-100, alpha=alpha)
                ax.scatter(
                    x, box_top + pathway_loc if self.grn_arr[i] > 0 else box_top, 
                    color=cmap[i], marker="v" if self.grn_arr[i] > 0 else "$-$"
                )

            for ii, j in enumerate(np.arange(i + 1, n_factor)):
                if self.adjacent_matrix.iloc[i, j] != 0:
                    x2, y2 = dat.x[j], dat.y[j]
                    if ii == 0:
                        ax.plot([x, x2], [y, y2], c=cmap[i], zorder=-100, alpha=alpha)
                        ax.scatter(
                            x2 - .39 if self.adjacent_matrix.iloc[i, j] > 0 else x2 - .33, 
                            y2 + .5 if self.adjacent_matrix.iloc[i, j] > 0 else y2 + .35, 
                            color=cmap[i], 
                            marker="^" if self.adjacent_matrix.iloc[i, j] > 0 else "$|$",
                            alpha=alpha
                        )
                    else:
                        h = y + ii + 0.2 * i + 1
                        ax.plot(
                            *fmt_curve((x,y), (x2,y2), h), 
                            zorder=-100, c=cmap[i], alpha=alpha
                        )
                        ax.scatter(
                            x2 - .15 if self.adjacent_matrix.iloc[i, j] > 0 else x2 - .1, 
                            y2 + .86 if self.adjacent_matrix.iloc[i, j] > 0 else y2 + .7, 
                            color=cmap[i], 
                            marker="<" if self.adjacent_matrix.iloc[i, j] > 0 else "$-$",
                            alpha=alpha
                        )

                    height = max(
                        height, 
                        get_top((x,y), (x2,y2), y + ii + 0.2 * i + 1, True)[1]
                    )

        ax.set_xlim(1.5, 1.5 + n_factor)
        ax.set_ylim(-3.0, height + 1)
        ax.axis("off");


def create_grn_arr(adj_mat: pd.DataFrame) -> np.ndarray:
    pathways = adj_mat.iloc[:, -1].values
    regulations =  adj_mat.iloc[:, :-1].values
    return np.hstack([
        pathways,
        regulations[np.triu_indices_from(regulations, k=1)]
    ])