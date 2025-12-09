import numpy as np


def random_grn_generator(n_factor: int, n_grn: int = 1, seed: int = 0):
    np.random.seed(seed)
    n_reg = int(n_factor * (n_factor - 1) / 2)
    return np.random.randint(-1, 2, (n_grn, n_factor + n_reg))
