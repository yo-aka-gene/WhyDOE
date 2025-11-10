import numpy as np

from doe_modules.preferences import harmonic_mean


def n_pathways(arr) -> int:
    return np.abs(arr)[:4].sum()


def n_edges(arr) -> int:
    return np.abs(arr)[4:].sum()


def pathway_coverage(arr) -> float:
    full = arr[:4].size
    return n_pathways(arr) / full


def pathway_positivity(arr):
    n_p = n_pathways(arr)
    return (arr[:4] == 1).sum() / n_p if n_p != 0 else 0


def pathway_negativity(arr):
    n_p = n_pathways(arr)
    return (arr[:4] == -1).sum() / n_p if n_p != 0 else 0


def positive_pathway_coverage(arr):
    return (arr[:4] == 1).sum() / arr[:4].size


def negative_pathway_coverage(arr):
    return (arr[:4] == -1).sum() / arr[:4].size


def sparse_pathway_coverage(arr):
    return (arr[:4] == 0).sum() / arr[:4].size


def edge_coverage(arr) -> float:
    full = arr[4:].size
    return n_edges(arr) / full


def n_pos(arr):
    return (arr[4:] == 1).sum()


def edge_positivity(arr):
    n_e = n_edges(arr)
    return n_pos(arr) / n_e if n_e != 0 else 0


def edge_negativity(arr):
    n_e = n_edges(arr)
    return (arr[4:] == -1).sum() / n_e if n_e != 0 else 0


def positive_edge_coverage(arr):
    return (arr[4:] == 1).sum() / arr[4:].size


def negative_edge_coverage(arr):
    return (arr[4:] == -1).sum() / arr[4:].size


def sparse_edge_coverage(arr):
    return (arr[4:] == 0).sum() / arr[4:].size


def is_effective(arr):
    return np.array([
        (arr[4] != 0) & ((arr[1] != 0) or (arr[(2, 7),] != 0).all() or (arr[(3, 8),] != 0).all() or (arr[(3, 7, 9),] != 0).all()),
        (arr[5] != 0) & ((arr[2] != 0) or (arr[(3, 9),] != 0).all()),
        (arr[6] != 0) & (arr[3] != 0),
        (arr[7] != 0) & ((arr[2] != 0) or (arr[(3, 9),] != 0).all()),
        (arr[8] != 0) & (arr[3] != 0),
        (arr[9] != 0) & (arr[3] != 0),
    ])


def n_eff(arr): 
    return is_effective(arr).sum()


def edge_effectivity(arr):
    n_e = n_edges(arr)
    return n_eff(arr) / n_e if n_e != 0 else 0


def n_effpos(arr):
    return np.array([
        (arr[4] > 0) & ((arr[1] != 0) or (arr[(2, 7),] != 0).all() or (arr[(3, 8),] != 0).all() or (arr[(3, 7, 9),] != 0).all()),
        (arr[5] > 0) & ((arr[2] != 0) or (arr[(3, 9),] != 0).all()),
        (arr[6] > 0) & (arr[3] != 0),
        (arr[7] > 0) & ((arr[2] != 0) or (arr[(3, 9),] != 0).all()),
        (arr[8] > 0) & (arr[3] != 0),
        (arr[9] > 0) & (arr[3] != 0),
    ]).sum()


def n_effneg(arr):
    return np.array([
        (arr[4] < 0) & ((arr[1] != 0) or (arr[(2, 7),] != 0).all() or (arr[(3, 8),] != 0).all() or (arr[(3, 7, 9),] != 0).all()),
        (arr[5] < 0) & ((arr[2] != 0) or (arr[(3, 9),] != 0).all()),
        (arr[6] < 0) & (arr[3] != 0),
        (arr[7] < 0) & ((arr[2] != 0) or (arr[(3, 9),] != 0).all()),
        (arr[8] < 0) & (arr[3] != 0),
        (arr[9] < 0) & (arr[3] != 0),
    ]).sum()


def effective_edge_positivity(arr):
    n_e = n_eff(arr)
    return n_effpos(arr) / n_e if n_e != 0 else 0


def effective_edge_negativity(arr):
    n_e = n_eff(arr)
    return n_effneg(arr) / n_e if n_e != 0 else 0

def edge_loading(arr):
    return np.abs(arr)[:4] * np.array([
        0,
        (arr[(4),] != 0).sum(),
        (arr[(5, 7),] != 0).sum() + (arr[(4, 7),] != 0).all(),
        (arr[(6, 8, 9),] != 0).sum() + (arr[(7, 9),] != 0).all() + (arr[(5, 9),] != 0).all() + ((arr[(4, 8),] != 0).all() or (arr[(4, 7, 9),] != 0).all()),
    ])

# def max_edge_loading(arr):
#     return  edge_loading(arr).max()

# def max_edge_density(arr):
#     return max_edge_loading(arr) / n_eff(arr) if n_eff(arr) != 0 else 0

# def mean_edge_density(arr):
#     return edge_loading(arr).mean() / n_eff(arr) if n_eff(arr) != 0 else 0

def max_edge_density(arr):
    return edge_loading(arr).max() / n_eff(arr) if n_eff(arr) != 0 else 0


def mean_edge_density(arr):
    n_factor = arr[:4].size
    theoretical = n_factor * (n_factor + 1) * (n_factor - 1) / 6
    return edge_loading(arr).sum() / theoretical if theoretical != 0 else 0


def positive_edge_loading(arr):
    return (arr[:4] == 1) * np.array([
        0,
        (arr[(4),] == 1).sum(),
        (arr[(5, 7),] == 1).sum() + (arr[(4, 7),] == 1).all(),
        (arr[(6, 8, 9),] == 1).sum() + (arr[(7, 9),] == 1).all() + (arr[(5, 9),] == 1).all() + ((arr[(4, 8),] == 1).all() or (arr[(4, 7, 9),] == 1).all()),
    ])


def max_positive_edge_density(arr):
    return positive_edge_loading(arr).max() / n_eff(arr) if n_eff(arr) != 0 else 0


def mean_positive_edge_density(arr):
    n_factor = arr[:4].size
    theoretical = n_factor * (n_factor + 1) * (n_factor - 1) / 6
    return positive_edge_loading(arr).sum() / theoretical if theoretical != 0 else 0


def synergetic_edge_loading(arr):
    return np.abs(arr)[:4] * np.array([
        0,
        (arr[(4),] == 1).sum(),
        (arr[(5, 7),] == 1).sum() + (arr[(4, 7),] == 1).all(),
        (arr[(6, 8, 9),] == 1).sum() + (arr[(7, 9),] == 1).all() + (arr[(5, 9),] == 1).all() + ((arr[(4, 8),] == 1).all() or (arr[(4, 7, 9),] == 1).all()),
    ])


def max_synergetic_edge_density(arr):
    return synergetic_edge_loading(arr).max() / n_eff(arr) if n_eff(arr) != 0 else 0


def mean_synergetic_edge_density(arr):
    n_factor = arr[:4].size
    theoretical = n_factor * (n_factor + 1) * (n_factor - 1) / 6
    return synergetic_edge_loading(arr).sum() / theoretical if theoretical != 0 else 0


def factor_loading(arr):
    return np.abs(arr)[:4] * np.array([
        1,
        1 + (arr[(4),] != 0).sum(),
        1 + (arr[(7),] != 0).sum() + ((arr[(4, 7),] != 0).all() or (arr[(5),] != 0).all()),
        1 + (arr[(9),] != 0).sum() + ((arr[(7, 9),] != 0).all() or (arr[(8),] != 0).all()) + \
        ((arr[(4, 7, 9),] != 0).all() or (arr[(4, 8),] != 0).all() or (arr[(5, 9),] != 0).all() or (arr[(6),] != 0).all()),
    ])


def max_factor_density(arr):
    return factor_loading(arr).max() / len(arr[:4])


# def mean_factor_density(arr):
#     return factor_loading(arr).mean() / len(arr[:4])

def mean_factor_density(arr):
    n_factor = arr[:4].size
    theoretical = n_factor * (n_factor + 1) / 2
    return factor_loading(arr).sum() / theoretical


def positive_factor_loading(arr):
    return (arr[:4] == 1) * np.array([
        1,
        1 + (arr[(4),] == 1).sum(),
        1 + (arr[(7),] == 1).sum() + ((arr[(4, 7),] == 1).all() or (arr[(5),] == 1).all()),
        1 + (arr[(9),] == 1).sum() + ((arr[(7, 9),] == 1).all() or (arr[(8),] == 1).all()) + \
        ((arr[(4, 7, 9),] == 1).all() or (arr[(4, 8),] == 1).all() or (arr[(5, 9),] == 1).all() or (arr[(6),] == 1).all()),
    ])


def max_positive_factor_density(arr):
    return positive_factor_loading(arr).max() / len(arr[:4])


def mean_positive_factor_density(arr):
    n_factor = arr[:4].size
    theoretical = n_factor * (n_factor + 1) / 2
    return positive_factor_loading(arr).sum() / theoretical


def synergetic_factor_loading(arr):
    return np.abs(arr)[:4] * np.array([
        1,
        1 + (arr[(4),] == 1).sum(),
        1 + (arr[(7),] == 1).sum() + ((arr[(4, 7),] == 1).all() or (arr[(5),] == 1).all()),
        1 + (arr[(9),] == 1).sum() + ((arr[(7, 9),] == 1).all() or (arr[(8),] == 1).all()) + \
        ((arr[(4, 7, 9),] == 1).all() or (arr[(4, 8),] == 1).all() or (arr[(5, 9),] == 1).all() or (arr[(6),] == 1).all()),
    ])

def max_synergetic_factor_density(arr):
    return synergetic_factor_loading(arr).max() / len(arr[:4])


def mean_synergetic_factor_density(arr):
    n_factor = arr[:4].size
    theoretical = n_factor * (n_factor + 1) / 2
    return synergetic_factor_loading(arr).sum() / theoretical


def cascade_length(arr):
    return np.abs(arr)[:4] * np.array([
        1,
        1 + np.max([0, (arr[(4),] != 0).sum()]),
        1 + np.max([
            0, (arr[(5),] != 0).sum(), (arr[(7),] != 0).sum(), 
            (arr[(4, 7),] != 0).sum() * (arr[(4, 7),] != 0).all()
        ]),
        1 + np.max([
            0, (arr[(6),] != 0).sum(), (arr[(8),] != 0).sum(), (arr[(9),] != 0).sum(), 
            (arr[(4, 8),] != 0).sum() * (arr[(4, 8),] != 0).all(), 
            (arr[(5, 9),] != 0).sum() * (arr[(5, 9),] != 0).all(), 
            (arr[(7, 9),] != 0).sum() * (arr[(7, 9),] != 0).all(),
            (arr[(4, 7, 9),] != 0).sum() * (arr[(4, 7, 9),] != 0).all()
        ]),
    ])


def max_cascade_length_ratio(arr):
    return cascade_length(arr).max() / arr[:4].size


def mean_cascade_length_ratio(arr):
    n_factor = arr[:4].size
    theoretical = (1 + arr[:4].size) * arr[:4].size / 2
    return cascade_length(arr).sum() / theoretical


def positive_cascade_length(arr):
    return (arr[:4] == 1) * np.array([
        1,
        1 + np.max([0, (arr[(4),] == 1).sum()]),
        1 + np.max([
            0, (arr[(5),] == 1).sum(), (arr[(7),] == 1).sum(), 
            (arr[(4, 7),] == 1).sum() * (arr[(4, 7),] == 1).all()
        ]),
        1 + np.max([
            0, (arr[(6),] == 1).sum(), (arr[(8),] == 1).sum(), (arr[(9),] == 1).sum(),
            (arr[(4, 8),] == 1).sum() * (arr[(4, 8),] == 1).all(), 
            (arr[(5, 9),] == 1).sum() * (arr[(5, 9),] == 1).all(), 
            (arr[(7, 9),] == 1).sum() * (arr[(7, 9),] == 1).all(),
            (arr[(4, 7, 9),] == 1).sum() * (arr[(4, 7, 9),] == 1).all()
        ]),
    ])


def max_positive_cascade_length_ratio(arr):
    return positive_cascade_length(arr).max() / arr[:4].size


def mean_positive_cascade_length_ratio(arr):
    n_factor = arr[:4].size
    theoretical = (1 + arr[:4].size) * arr[:4].size / 2
    return positive_cascade_length(arr).sum() / theoretical


def synergetic_cascade_length(arr):
    return np.abs(arr)[:4] * np.array([
        1,
        1 + np.max([0, (arr[(4),] == 1).sum()]),
        1 + np.max([
            0, (arr[(5),] == 1).sum(), (arr[(7),] == 1).sum(), 
            (arr[(4, 7),] == 1).sum() * (arr[(4, 7),] == 1).all()
        ]),
        1 + np.max([
            0, (arr[(6),] == 1).sum(), (arr[(8),] == 1).sum(), (arr[(9),] == 1).sum(),
            (arr[(4, 8),] == 1).sum() * (arr[(4, 8),] == 1).all(), 
            (arr[(5, 9),] == 1).sum() * (arr[(5, 9),] == 1).all(), 
            (arr[(7, 9),] == 1).sum() * (arr[(7, 9),] == 1).all(),
            (arr[(4, 7, 9),] == 1).sum() * (arr[(4, 7, 9),] == 1).all()
        ]),
    ])


def max_synergetic_cascade_length_ratio(arr):
    return synergetic_cascade_length(arr).max() / arr[:4].size


def mean_synergetic_cascade_length_ratio(arr):
    n_factor = arr[:4].size
    theoretical = (1 + arr[:4].size) * arr[:4].size / 2
    return synergetic_cascade_length(arr).sum() / theoretical


def cascade_coverage(arr):
    return np.sum([
        (arr[(0),] != 0).all(),
        (arr[(4, 1),] != 0).all(),
        (arr[(4, 7, 2),] != 0).all(),
        (arr[(4, 7, 9, 3),] != 0).all(),
        (arr[(4, 8, 3),] != 0).all(),
        (arr[(5, 2),] != 0).all(),
        (arr[(5, 9, 3),] != 0).all(),
        (arr[(6, 3),] != 0).all(),
        (arr[(1),] != 0).all(),
        (arr[(7, 2),] != 0).all(),
        (arr[(7, 9, 3),] != 0).all(),
        (arr[(8, 3),] != 0).all(),
        (arr[(2),] != 0).all(),
        (arr[(9, 3),] != 0).all(),
        (arr[(3),] != 0).all(),
    ]) / (2 ** arr[:4].size - 1)


def positive_cascade_coverage(arr):
    return np.sum([
        (arr[(0),] == 1).all(),
        (arr[(4, 1),] == 1).all(),
        (arr[(4, 7, 2),] == 1).all(),
        (arr[(4, 7, 9, 3),] == 1).all(),
        (arr[(4, 8, 3),] == 1).all(),
        (arr[(5, 2),] == 1).all(),
        (arr[(5, 9, 3),] == 1).all(),
        (arr[(6, 3),] == 1).all(),
        (arr[(1),] == 1).all(),
        (arr[(7, 2),] == 1).all(),
        (arr[(7, 9, 3),] == 1).all(),
        (arr[(8, 3),] == 1).all(),
        (arr[(2),] == 1).all(),
        (arr[(9, 3),] == 1).all(),
        (arr[(3),] == 1).all(),
    ]) / (2 ** arr[:4].size - 1)


def synergetic_cascade_coverage(arr):
    return np.sum([
        (arr[(0),] != 0).all(),
        (arr[(4),] == 1).all() & (arr[1] != 0),
        (arr[(4, 7),] == 1).all() & (arr[2] != 0),
        (arr[(4, 7, 9),] == 1).all() & (arr[3] != 0),
        (arr[(4, 8),] == 1).all() & (arr[3] != 0),
        (arr[(5),] == 1).all() & (arr[2] != 0),
        (arr[(5, 9),] == 1).all() & (arr[3] != 0),
        (arr[(6),] == 1).all() & (arr[3] != 0),
        (arr[(1),] != 0).all(),
        (arr[(7),] == 1).all() & (arr[2] != 0),
        (arr[(7, 9),] == 1).all() & (arr[3] != 0),
        (arr[(8),] == 1).all() & (arr[3] != 0),
        (arr[(2),] != 0).all(),
        (arr[(9),] == 1).all() & (arr[3] != 0),
        (arr[(3),] != 0).all(),
    ]) / (2 ** arr[:4].size - 1)


def cai(arr) -> float:
    return harmonic_mean(
        edge_coverage(arr),
        sparse_pathway_coverage(arr),
        max_positive_edge_density(arr)
    )
