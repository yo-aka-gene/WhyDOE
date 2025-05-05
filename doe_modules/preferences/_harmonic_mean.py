import numpy as np


def harmonic_mean(*args):
    n = len(args)
    return n / (np.ones(n) / np.array(args)).sum() if (np.array(args) != 0).all() else 0
