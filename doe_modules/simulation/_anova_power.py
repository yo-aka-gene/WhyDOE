from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import f, ncf
import statsmodels.api as sm

from ._abstract import AbstractSimulator
from ._mlr import MLR
from doe_modules.design import DesignMatrix


def ff_anova_power(
    simulator: AbstractSimulator,
    mlr: MLR,
    typ: int = 2,
    alpha: float = 0.05
):
    """
    performs power analysis (n-th-order interaction terms at n>2 are not supported)
    """
    assert simulator.metadata["design"] == mlr.metadata["design"], \
        f"pass the same simulator that was also used for mlr"
    assert simulator.metadata["n_rep"] == mlr.metadata["n_rep"], \
        f"pass the same simulator that was also used for mlr"
    assert simulator.metadata["kwargs"] == mlr.metadata["kwargs"], \
        f"pass the same simulator that was also used for mlr"
    assert simulator.metadata["design"] == "FF", \
        f"only balanced ANOVA is currently supported"

    exmatrix = simulator.exmatrix
    y = simulator.exresult
    n_run = len(simulator.exmatrix)
    n_rep = simulator.metadata["n_rep"]
    n_factor = simulator.n_factor
    interaction_exists = mlr.metadata["interactions"]
    indices = simulator.exmatrix.iloc[:int(n_run / n_rep), :].index

    df_anova = sm.stats.anova_lm(mlr.result, typ=typ)

    # estimating sigma
    y_ijk = pd.concat([
        pd.DataFrame(
            y,
            index=exmatrix.index,
            columns=["y"]
        ).reset_index().groupby("index").mean().loc[indices]
    ] * n_rep).values.ravel()

    sigma = np.std(np.sqrt(n_rep / (n_rep - 1)) * (y - y_ijk), ddof=1)

    # estimating main effects
    # y_i: y_i values for main effects at each level of Xi
    # coef: coefficients of MLR model (main effects at xi = 1)
    y_i = np.vstack([
        [
            y[np.where(exmatrix[v] == 1)].mean(),
            y[np.where(exmatrix[v] == -1)].mean()
        ] for v in exmatrix
    ])

    y_bar = y.mean()
    main = y_i - y_bar

    df_main = pd.DataFrame(
        {
            "coef": y_i[:, 0] - y_bar,
            "std": sigma * np.ones(n_factor),
            "dfn": df_anova.df[exmatrix.columns],
            "dfd": df_anova.df["Residual"] * np.ones(n_factor),
            "nc": n_run * ((main / sigma) ** 2).mean(axis=1)
        },
        index=exmatrix.columns
    )

    if interaction_exists:
        order = mlr.metadata["order"]
        full_model = mlr.metadata["full_model"]

        interaction_names = ["".join(xij) for xij in combinations(exmatrix.columns, 2)]

        signs = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        cols = [(lambda x, y: (int((x - 1) / 2), int((y - 1) / 2)))(*s) for s in signs]

        # estimating interaction effects
        # y_ij: y_ij values for interaction effects at each level of Xi and Xj
        # coef: coefficients of MLR model (interaction effects at xi = 1 and xj=1)
        y_ij = np.hstack([
            np.array([
                y[np.where((exmatrix[xi] == si) & (exmatrix[xj] == sj))].mean() for (xi, xj) in combinations(exmatrix.columns, 2)
            ]).reshape(-1, 1) for (si, sj) in signs
        ])

        y_i_y_j = np.hstack([
            np.array([
                y_i[idxij, cij].sum() for idxij in combinations(np.arange(n_factor), 2)
            ]).reshape(-1, 1) for cij in cols
        ])

        interaction = y_ij - y_i_y_j + y_bar

        df_interaction = pd.DataFrame(
            {
                "coef": interaction[:, 0],
                "std": sigma * np.ones(len(interaction)),
                "dfn": df_anova.df[interaction_names],
                "dfd": df_anova.df["Residual"] * np.ones(len(interaction)),
                "nc": n_run * ((interaction / sigma) ** 2).mean(axis=1)
            },
            index=interaction_names
        )

    df = pd.concat([df_main, df_interaction]) if interaction_exists else df_main
    return df.assign(
        power=lambda d: ncf.sf(
            x=f.isf(alpha, dfn=d.dfn, dfd=d.dfd), 
            dfn=d.dfn, 
            dfd=d.dfd, 
            nc=d.nc
        )
    )
