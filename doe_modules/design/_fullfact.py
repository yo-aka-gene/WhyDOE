from itertools import product
import numpy as np

from ._abstract import DesignMatrix, DOE


class FullFactorial(DOE):
    def __init__(self, name: str = "Full factorial"):
        super().__init__(name=name)

    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        super().get_exmatrix(n_factor=n_factor)
        return DesignMatrix(
            np.array(
                list(product(*[[-1, 1] for i in range(n_factor)]))
            )
        )


    def __call__(self):
        return super().__call__()