from itertools import product
import numpy as np

from ._abstract import DesignMatrix, DOE


class FullFactorial(DOE):
    def __init__(self):
        super().__init__()

    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        return DesignMatrix(
            np.array(
                list(product(*[[-1, 1] for i in range(n_factor)]))
            )
        )