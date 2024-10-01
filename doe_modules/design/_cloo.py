import numpy as np

from ._abstract import DesignMatrix, DOE


class CLOO(DOE):
    def __init__(self, name: str = "C+LOO"):
        super().__init__(name=name)

    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        super().get_exmatrix(n_factor=n_factor)
        res = np.vstack([
            np.ones(n_factor),
            np.ones((n_factor, n_factor)) - 2 * np.eye(n_factor),
        ])
        return DesignMatrix(res)


    def __call__(self):
        return super().__call__()


class YCLOO(DOE):
    def __init__(self, name: str = "Y-C+LOO"):
        super().__init__(name=name)

    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        super().get_exmatrix(n_factor=n_factor)
        res = np.vstack([
            np.ones(n_factor),
            np.ones((n_factor, n_factor)) - 2 * np.eye(n_factor),
            -np.ones(n_factor),
            2 * np.eye(n_factor) - np.ones((n_factor, n_factor)),
        ])
        return DesignMatrix(res)
    

    def __call__(self):
        return super().__call__()
