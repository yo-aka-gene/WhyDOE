import joblib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns


def n_factor(arr) -> int:
    return int((np.sqrt(1 + 8 * arr.size) - 1) / 2)  


def n_pathways(arr) -> int:
    n = n_factor(arr)
    return np.abs(arr)[:n].sum()


def n_edges(arr) -> int:
    n = n_factor(arr)
    return np.abs(arr)[n:].sum()


def pathway_coverage(arr) -> float:
    n = n_factor(arr)
    return n_pathways(arr) / n


def pathway_positivity(arr):
    n_f = n_factor(arr)
    n_p = n_pathways(arr)
    return (arr[:n_f] == 1).sum() / n_p if n_p != 0 else 0


def pathway_negativity(arr):
    n_f = n_factor(arr)
    n_p = n_pathways(arr)
    return (arr[:n_f] == -1).sum() / n_p if n_p != 0 else 0


def positive_pathway_coverage(arr):
    n_f = n_factor(arr)
    return (arr[:n_f] == 1).sum() / n_f


def negative_pathway_coverage(arr):
    n_f = n_factor(arr)
    return (arr[:n_f] == -1).sum() / n_f


def sparse_pathway_coverage(arr):
    n_f = n_factor(arr)
    return (arr[:n_f] == 0).sum() / n_f


def edge_coverage(arr) -> float:
    n_f = n_factor(arr)
    full = arr[n_f:].size
    return n_edges(arr) / full


def n_pos(arr):
    n_f = n_factor(arr)
    return (arr[n_f:] == 1).sum()


def edge_positivity(arr):
    n_e = n_edges(arr)
    return n_pos(arr) / n_e if n_e != 0 else 0


def edge_negativity(arr):
    n_f = n_factor(arr)
    n_e = n_edges(arr)
    return (arr[n_f:] == -1).sum() / n_e if n_e != 0 else 0


def positive_edge_coverage(arr):
    n_f = n_factor(arr)
    return (arr[n_f:] == 1).sum() / arr[n_f:].size


def negative_edge_coverage(arr):
    n_f = n_factor(arr)
    return (arr[n_f:] == -1).sum() / arr[n_f:].size


def sparse_edge_coverage(arr):
    n_f = n_factor(arr)
    return (arr[n_f:] == 0).sum() / arr[n_f:].size


def effectivity_matrix(arr, bool_encoder = lambda x: x != 0):
    n_f = n_factor(arr)    
    mat = []
    operator = np.eye(n_f).astype(bool)
    
    for nrow, idx in enumerate(np.lib.stride_tricks.sliding_window_view(
        np.arange(n_f, 0, -1).cumsum(), 
        window_shape=2
    )):
        _edge = arr[idx[0]:idx[1]]
        diag = bool_encoder(np.diag(_edge))
        mat += [
            np.hstack([
                np.zeros((_edge.size, (n_f - _edge.size))).astype(bool),
                diag
            ])
        ]
        operator[nrow, nrow + 1:] = bool_encoder(_edge)
    
    connection = np.vstack(mat)
    
    for i in range(n_f):
        connection @= operator
        new_row = np.zeros(n_f).astype(bool)
        new_row[i] = True
        operator[i, :] = new_row
    
    return connection


def is_effective(arr):
    n_f = n_factor(arr)
    n_reg = int(n_f * (n_f - 1) / 2)
    bool_encoder = lambda x: x != 0
    pathways = np.vstack([bool_encoder(arr[:n_f])] * n_reg)
    mat = effectivity_matrix(arr, bool_encoder=bool_encoder)
    return (pathways * mat).sum(axis=1).astype(bool)


def n_eff(arr):
    return is_effective(arr).sum()


def edge_effectivity(arr):
    n_e = n_edges(arr)
    return n_eff(arr) / n_e if n_e != 0 else 0


def n_effpos(arr):
    n_f = n_factor(arr)
    return ((arr[n_f:] == 1) & is_effective(arr)).sum()


def n_effneg(arr):
    n_f = n_factor(arr)
    return ((arr[n_f:] == -1) & is_effective(arr)).sum()


def effective_edge_positivity(arr):
    n_e = n_eff(arr)
    return n_effpos(arr) / n_e if n_e != 0 else 0


def effective_edge_negativity(arr):
    n_e = n_eff(arr)
    return n_effneg(arr) / n_e if n_e != 0 else 0


def edge_loading(arr):
    n_f = n_factor(arr)
    n_reg = int(n_f * (n_f - 1) / 2)
    bool_encoder = lambda x: x != 0
    pathways = np.vstack([bool_encoder(arr[:n_f])] * n_reg)
    mat = effectivity_matrix(arr, bool_encoder=bool_encoder)
    return (pathways * mat).sum(axis=0)


def max_edge_density(arr):
    return edge_loading(arr).max() / n_eff(arr) if n_eff(arr) != 0 else 0


def mean_edge_density(arr):
    n_f = n_factor(arr)
    theoretical = n_f * (n_f + 1) * (n_f - 1) / 6
    return edge_loading(arr).sum() / theoretical if theoretical != 0 else 0


def positive_edge_loading(arr):
    n_f = n_factor(arr)
    n_reg = int(n_f * (n_f - 1) / 2)
    bool_encoder = lambda x: x == 1
    pathways = np.vstack([bool_encoder(arr[:n_f])] * n_reg)
    mat = effectivity_matrix(arr, bool_encoder=bool_encoder)
    return (pathways * mat).sum(axis=0)


def max_positive_edge_density(arr):
    return positive_edge_loading(arr).max() / n_eff(arr) if n_eff(arr) != 0 else 0


def mean_positive_edge_density(arr):
    n_f = n_factor(arr)
    theoretical = n_f * (n_f + 1) * (n_f - 1) / 6
    return positive_edge_loading(arr).sum() / theoretical if theoretical != 0 else 0


def synergetic_edge_loading(arr):
    n_f = n_factor(arr)
    n_reg = int(n_f * (n_f - 1) / 2)
    pathways = np.vstack([(lambda x: x != 0)(arr[:n_f])] * n_reg)
    mat = effectivity_matrix(arr, bool_encoder=lambda x: x == 1)
    return (pathways * mat).sum(axis=0)


def max_synergetic_edge_density(arr):
    return synergetic_edge_loading(arr).max() / n_eff(arr) if n_eff(arr) != 0 else 0


def mean_synergetic_edge_density(arr):
    n_f = n_factor(arr)
    theoretical = n_f * (n_f + 1) * (n_f - 1) / 6
    return synergetic_edge_loading(arr).sum() / theoretical if theoretical != 0 else 0


def factor_loading(arr):
    n_f = n_factor(arr)
    connection = effectivity_matrix(arr)
    
    mat = []

    for nrow, idx in enumerate(np.lib.stride_tricks.sliding_window_view(
        np.hstack([[0], np.arange(n_f - 1, 0, -1).cumsum()]), 
        window_shape=2
    )):
        mat += [connection[idx[0]:idx[1]].sum(axis=0).astype(bool)]

    return (
        np.vstack(mat).sum(axis=0) + 1
    ) * (arr[:n_f] != 0).astype(int)


def max_factor_density(arr):
    return factor_loading(arr).max() / n_factor(arr)


def mean_factor_density(arr):
    n_f = n_factor(arr)
    theoretical = n_f * (n_f + 1) / 2
    return factor_loading(arr).sum() / theoretical


def positive_factor_loading(arr):
    n_f = n_factor(arr)
    connection = effectivity_matrix(arr, bool_encoder=lambda x: x == 1)
    
    mat = []

    for nrow, idx in enumerate(np.lib.stride_tricks.sliding_window_view(
        np.hstack([[0], np.arange(n_f - 1, 0, -1).cumsum()]), 
        window_shape=2
    )):
        mat += [connection[idx[0]:idx[1]].sum(axis=0).astype(bool)]

    return (
        np.vstack(mat).sum(axis=0) + 1
    ) * (arr[:n_f] == 1).astype(int)


def max_positive_factor_density(arr):
    return positive_factor_loading(arr).max() / n_factor(arr)


def mean_positive_factor_density(arr):
    n_f = n_factor(arr)
    theoretical = n_f * (n_f + 1) / 2
    return positive_factor_loading(arr).sum() / theoretical


def synergetic_factor_loading(arr):
    n_f = n_factor(arr)
    connection = effectivity_matrix(arr, bool_encoder=lambda x: x == 1)
    
    mat = []

    for nrow, idx in enumerate(np.lib.stride_tricks.sliding_window_view(
        np.hstack([[0], np.arange(n_f - 1, 0, -1).cumsum()]), 
        window_shape=2
    )):
        mat += [connection[idx[0]:idx[1]].sum(axis=0).astype(bool)]

    return (
        np.vstack(mat).sum(axis=0) + 1
    ) * (arr[:n_f] != 0).astype(int)


def max_synergetic_factor_density(arr):
    return synergetic_factor_loading(arr).max() / n_factor(arr)


def mean_synergetic_factor_density(arr):
    n_f = n_factor(arr)
    theoretical = n_f * (n_f + 1) / 2
    return synergetic_factor_loading(arr).sum() / theoretical


def cascade_length(arr):
    n_f = n_factor(arr)
    
    adj = np.zeros((n_f, n_f), dtype=int)
    rows, cols = np.triu_indices(n_f, k=1)
    adj[rows, cols] = (arr[n_f:] != 0)
    
    L = np.ones(n_f, dtype=int)
    
    for i in range(1, n_f):
        incoming = adj[:i, i]
        current_max = np.max(L[:i] * incoming)
        L[i] = 1 + current_max

    return L * (arr[:n_f] != 0).astype(int)


def max_cascade_length_ratio(arr):
    return cascade_length(arr).max() / n_factor(arr)


def mean_cascade_length_ratio(arr):
    n_f = n_factor(arr)
    theoretical = (1 + n_f) * n_f / 2
    return cascade_length(arr).sum() / theoretical


def positive_cascade_length(arr):
    n_f = n_factor(arr)
    
    adj = np.zeros((n_f, n_f), dtype=int)
    rows, cols = np.triu_indices(n_f, k=1)
    adj[rows, cols] = (arr[n_f:] == 1)
    
    L = np.ones(n_f, dtype=int)
    
    for i in range(1, n_f):
        incoming = adj[:i, i]
        current_max = np.max(L[:i] * incoming)
        L[i] = 1 + current_max

    return L * (arr[:n_f] == 1).astype(int)


def max_positive_cascade_length_ratio(arr):
    return positive_cascade_length(arr).max() / n_factor(arr)


def mean_positive_cascade_length_ratio(arr):
    n_f = n_factor(arr)
    theoretical = (1 + n_f) * n_f / 2
    return positive_cascade_length(arr).sum() / theoretical


def synergetic_cascade_length(arr):
    n_f = n_factor(arr)
    
    adj = np.zeros((n_f, n_f), dtype=int)
    rows, cols = np.triu_indices(n_f, k=1)
    adj[rows, cols] = (arr[n_f:] == 1)
    
    L = np.ones(n_f, dtype=int)
    
    for i in range(1, n_f):
        incoming = adj[:i, i]
        current_max = np.max(L[:i] * incoming)
        L[i] = 1 + current_max

    return L * (arr[:n_f] != 0).astype(int)


def max_synergetic_cascade_length_ratio(arr):
    return synergetic_cascade_length(arr).max() / n_factor(arr)


def mean_synergetic_cascade_length_ratio(arr):
    n_f = n_factor(arr)
    theoretical = (1 + n_f) * n_f / 2
    return synergetic_cascade_length(arr).sum() / theoretical


def pbsi(arr) -> float:
    return np.mean([
        positive_pathway_coverage(arr),
        1 - max_positive_edge_density(arr),
        1 - mean_positive_edge_density(arr)
    ])


feat_names_short = dict(
    pathway_coverage=r"P%",
    pathway_positivity=r"P$_{(+)}$/P",
    pathway_negativity=r"P$_{(-)}$/P",
    positive_pathway_coverage=r"P$_{(+)}$%",
    negative_pathway_coverage=r"P$_{(-)}$%",
    sparse_pathway_coverage=r"P$_{(0)}$%",
    edge_coverage=r"R%",
    edge_positivity=r"R$_{(+)}$/R",
    edge_negativity=r"R$_{(-)}$/R",
    positive_edge_coverage=r"R$_{(+)}$%",
    negative_edge_coverage=r"R$_{(-)}$%",
    sparse_edge_coverage=r"R$_{(0)}$%",
    edge_effectivity=r"R$^*$/R",
    effective_edge_positivity=r"R$^*_{(+)}$/R$^*$",
    effective_edge_negativity=r"R$^*_{(-)}$/R$^*$",
    max_edge_density=r"MaxRW/R$^*$",
    mean_edge_density=r"RW%",
    max_positive_edge_density=r"MaxRW$_{(+)}$/R$^*$",
    mean_positive_edge_density=r"RW$_{(+)}$%",
    max_synergetic_edge_density=r"MaxRW$_&$/R$^*$",
    mean_synergetic_edge_density=r"RW$_&$%",
    max_factor_density=r"MaxFW/n",
    mean_factor_density=r"FW%",
    max_positive_factor_density=r"MaxFW$_{(+)}$/n",
    mean_positive_factor_density=r"FW$_{(+)}$%",
    max_synergetic_factor_density=r"MaxFW$_&$/n",
    mean_synergetic_factor_density=r"FW$_&$%",
    max_cascade_length_ratio=r"MaxCL/n",
    mean_cascade_length_ratio=r"CL%",
    max_positive_cascade_length_ratio=r"MaxCL$_{(+)}$/n",
    mean_positive_cascade_length_ratio=r"CL$_{(+)}$%",
    max_synergetic_cascade_length_ratio=r"MaxCL$_&$/n",
    mean_synergetic_cascade_length_ratio=r"CL$_&$%"
)


pbsi_features = [
    'positive_pathway_coverage',
    'max_positive_edge_density',
    'mean_positive_edge_density',
]


negs = [
    "max_positive_edge_density", 
    "max_positive_cascade_length_ratio", 
    "max_synergetic_edge_density",
    "mean_positive_edge_density",
    "effective_edge_positivity"
]


top10 = np.array([
    "positive_pathway_coverage", 
    "max_positive_edge_density", 
    "max_positive_cascade_length_ratio", 
    "edge_effectivity",
    "mean_factor_density",
    "max_synergetic_edge_density",
    "mean_positive_edge_density",
    "pathway_coverage",
    "max_cascade_length_ratio",
    "effective_edge_positivity"
])


grn_cmap = joblib.load(Path(__file__).resolve().parent / "grn_cmap.joblib")


def plot_grn_metrics(
    arr: np.ndarray,
    ax: plt.Axes
):
    df_network_metrics = pd.DataFrame({
        "scores": [
            positive_pathway_coverage(arr), 
            1 - max_positive_edge_density(arr),
            1 - mean_positive_edge_density(arr),
            pbsi(arr)
        ],
        "": [
            "$1-$" + feat_names_short[k] if k in negs else feat_names_short[k] for k in pbsi_features
        ] + ["PBSI"]
    })
    
    sns.scatterplot(
        data=df_network_metrics,
        y="", x="scores", palette=[grn_cmap[k] for k in pbsi_features + ["pbsi"]],
        hue="", legend=False, s=200
    )
    
    for i, c in enumerate([grn_cmap[k] for k in pbsi_features + ["pbsi"]]):
        score = df_network_metrics.scores[i]
        ax.hlines(i, 0, score, color=c, lw=3)
        ax.text(
            score -.05 if score > .5 else score + 0.1,
            i - 0.05 if score > .5 else i,
            df_network_metrics[""][i], c=c, 
            ha="right" if score > .5 else "left", 
            va="bottom" if score > .5 else "center"
        )
    
    ylim = ax.get_ylim()
    ax.vlines(0, *ylim, color=".2", lw=.5, zorder=-10)
    ax.vlines(1, *ylim, color=".2", lw=.5, zorder=-10)
    
    [ax.spines[loc].set_visible(False) for loc in ["top", "right", "left"]]
    ax.tick_params(labelleft=False, left=False)
    
    ax.set(xlim=(-.05, 1.05), ylim=ylim, xlabel="Index values")