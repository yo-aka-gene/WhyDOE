import numpy as np

from ._abstract import DesignMatrix, DOE


class Yamanaka(DOE):
    def __init__(self):
        super().__init__()

    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        res = np.vstack([
            np.ones(n_factor),
            np.ones((n_factor, n_factor)) - 2 * np.eye(n_factor),
        ])
        return DesignMatrix(res)