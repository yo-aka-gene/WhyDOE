import numpy as np
from pyDOE import pbdesign

from ._abstract import DesignMatrix, DOE


class PlackettBurman(DOE):
    def __init__(self):
        super().__init__()

    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        return DesignMatrix(pbdesign(n_factor))