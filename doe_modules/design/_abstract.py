from itertools import combinations

import numpy as np
import pandas as pd


class DesignMatrix:
    def __init__(self, matrix: np.ndarray):
        self.matrix = pd.DataFrame(
            matrix,
            index = [f"#{i + 1}" for i in range(matrix.shape[0])],
            columns = [f"X{i + 1}" for i in range(matrix.shape[1])]
        )
        self.shape = matrix.shape
        self.values = matrix


    def __call__(self, encode: bool = False):
        return ((self.matrix + 1) / 2).astype(bool) if encode else self.matrix


    def interactions(self, order: int = 2):
        return pd.concat(
            [
                pd.DataFrame({
                    "".join(v): self.matrix.loc[
                        :, v
                    ].prod(axis=1) for v in combinations(
                        self.matrix.columns, n
                    )
                }) for n in np.arange(
                    1, 
                    self.matrix.columns.size + 1 if order == np.inf else order + 1
                )
            ],
            axis=1
        )


class DOE:
    def __init__(self, name: str):
        self.is_initialized = True
        self.name = name


    def get_exmatrix(
        self, 
        n_factor: int
    ) -> DesignMatrix:
        self.title = f"{self.name} design (n={n_factor})"


    def __call__(self):
        return self
