import numpy as np
from pyDOE import pbdesign

from ._abstract import DesignMatrix, DOE


class PlackettBurman(DOE):
    def __init__(self, name: str = "PB"):
        super().__init__(name=name)

    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        super().get_exmatrix(n_factor=n_factor)
        return DesignMatrix(pbdesign(n_factor))


    def __call__(self):
        return super().__call__()